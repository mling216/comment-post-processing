"""
Claude API Visual Complexity (VC) Scoring Script
=================================================
Sends each image to Claude with a few-shot anchored prompt and receives
7-dimension VC scores (0–1) plus an overall vc_score.

Images can be loaded from local files or fetched from URLs.
Outputs are CSV-compatible with the existing vc_scores.csv / vc_explanations.csv format.

Usage:
    python _vc_score_api_v3.py                     # process all images from mapping CSV
    python _vc_score_api_v3.py --limit 5           # process first N images
    python _vc_score_api_v3.py --overwrite         # re-process already scored images
    python _vc_score_api_v3.py --source local      # use local image files
    python _vc_score_api_v3.py --source url        # fetch images from URLs (default)
    python _vc_score_api_v3.py --outdir results    # custom output directory
"""

import os, sys, json, time, argparse, base64, re, csv, asyncio, threading
from pathlib import Path
from io import BytesIO
import anthropic
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

MODEL          = 'claude-sonnet-4-6'
MAX_TOKENS     = 1500
SLEEP_BETWEEN  = 0.5        # seconds between API calls

MAPPING_CSV    = Path(__file__).parent.parent / 'phrase_reduction_v2' / 'image_phrase_word_mapping.csv'
LOCAL_IMG_DIR  = Path(__file__).parent.parent / 'Claude_vc_prediction' / 'images'
DEFAULT_OUTDIR = Path(__file__).parent.parent / 'vc_api_scores'

DIMENSIONS = [
    'data_density', 'visual_encoding', 'text_annotation',
    'domain_schema', 'color_symbol', 'aesthetic_order', 'cognitive_load'
]

# ── Anchor images (few-shot examples) ──────────────────────────────────────
# These are embedded as conversation examples so the model is calibrated.
ANCHORS = [
    {
        'imageName': 'VisC.503.6.png',
        'vc_score': 0.22,
        'scores': {
            'data_density': 0.15,
            'visual_encoding': 0.10,
            'text_annotation': 0.20,
            'domain_schema': 0.15,
            'color_symbol': 0.10,
            'aesthetic_order': 0.10,
            'cognitive_load': 0.20,
        },
        'explanations': {
            'data_density': 'Only 3 bars with error bars; very sparse data content.',
            'visual_encoding': 'Standard vertical bars with simple hatching; minimal encoding variety.',
            'text_annotation': 'Basic axis labels and a title present but minimal text overall.',
            'domain_schema': 'Generic bar chart format; no specialized domain knowledge needed.',
            'color_symbol': 'Monochrome black hatched bars on white; no color variety at all.',
            'aesthetic_order': 'Clean, well-organized layout with ample whitespace.',
            'cognitive_load': 'Immediately interpretable; very low effort to understand.',
        },
        'summary': 'A simple 3-bar chart with black hatching and error bars. Extremely low visual complexity across all dimensions.'
    },
    {
        'imageName': 'InfoVisJ.619.17.png',
        'vc_score': 0.54,
        'scores': {
            'data_density': 0.55,
            'visual_encoding': 0.50,
            'text_annotation': 0.45,
            'domain_schema': 0.60,
            'color_symbol': 0.50,
            'aesthetic_order': 0.45,
            'cognitive_load': 0.55,
        },
        'explanations': {
            'data_density': 'Moderate number of data points in a 2D scatterplot with some grouping structure visible.',
            'visual_encoding': 'Points plotted in a biplot layout with directional arrows; moderate encoding complexity.',
            'text_annotation': 'Axis labels and some point labels present; moderate text volume.',
            'domain_schema': 'Biplot/PCA projection requires some statistical familiarity.',
            'color_symbol': 'A few colors distinguish groups; moderate palette diversity.',
            'aesthetic_order': 'Reasonably organized but arrows and overlapping labels add mild clutter.',
            'cognitive_load': 'Requires understanding of biplots and PCA; moderate interpretation effort.',
        },
        'summary': 'A PCA biplot with directional arrows and grouped points. Moderate complexity requiring some statistical knowledge.'
    },
    {
        'imageName': 'InfoVisJ.1149.6(1).png',
        'vc_score': 0.95,
        'scores': {
            'data_density': 0.90,
            'visual_encoding': 0.85,
            'text_annotation': 0.80,
            'domain_schema': 0.95,
            'color_symbol': 0.70,
            'aesthetic_order': 0.85,
            'cognitive_load': 0.95,
        },
        'explanations': {
            'data_density': 'Multiple coordinated panels (7 views) each showing dense text and data; extremely high information volume.',
            'visual_encoding': 'Mixed encoding types across panels: lists, highlighted text, bar-like indicators, tag clouds; high variety.',
            'text_annotation': 'Extensive text content across all panels; the visualization IS largely text-based with highlights and annotations.',
            'domain_schema': 'Specialized text analysis/collation tool; requires understanding of multi-view coordinated systems.',
            'color_symbol': 'Multiple highlight colors for text segments; moderate color use with functional meaning.',
            'aesthetic_order': 'Complex multi-panel layout with varied panel sizes; visually dense and somewhat overwhelming.',
            'cognitive_load': 'Must track relationships across 7 coordinated views with dense text; very high cognitive demand.',
        },
        'summary': 'A multi-panel text collation tool with 7 coordinated views showing highlighted text passages, bar indicators, and tag clouds. Extremely high visual complexity requiring sustained multi-view comparison.'
    },
]

# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a visual complexity (VC) scoring expert for data visualizations.

**Visual Complexity** measures the amount of detail, intricacy, and perceptual/cognitive demand of a visualization image. It is scored on a 0–1 scale where 0 = minimal complexity and 1 = maximum complexity.

You will receive a single visualization image with NO accompanying text or comments. Score it purely from what you see.

## Calibration Guidance
Use the full 0–1 range. Do NOT cluster scores conservatively toward the center.
- A plain single bar chart or pie chart with < 5 elements → vc_score ≈ 0.25–0.40
- A standard scatterplot, grouped bar chart, or line chart → vc_score ≈ 0.45–0.60
- A multi-panel dashboard, dense network, or multi-encoding composite → vc_score ≈ 0.70–0.85
- An extremely dense multi-view system with many coordinated panels → vc_score ≈ 0.85–0.95
IMPORTANT: If you are unsure between two scores, always choose the higher one. Most scorers underestimate — push scores up.

## Scoring Dimensions (each 0–1)

1. **data_density** — The perceived amount, richness, or depth of data content. Considers information volume, element quantity, and visual clutter/overlap.
2. **visual_encoding** — The variety, type, and complexity of graphical forms (shapes, lines, marks) and how spatial layout, scale, and encoding interpretability contribute to complexity.
3. **text_annotation** — The sheer quantity and density of text elements (titles, axis labels, legends, captions, annotations, in-chart labels). Score based on volume only: more text = higher score, regardless of legibility.
4. **domain_schema** — Whether specialized domain knowledge is needed, including dimensionality (2D/3D), structural complexity, and abstraction level.
5. **color_symbol** — Range, variety, and arrangement of colors, plus use of symbols, textures, and non-color graphical markers.
6. **aesthetic_order** — How visually cluttered, dense, or disordered the layout appears. Higher = more cluttered/overwhelming. A clean minimal layout scores low; a crowded layout with overlapping elements scores high.
7. **cognitive_load** — Overall ease or difficulty of interpreting the visualization. Considers interpretive difficulty, semantic clarity, and processing time/effort.

## Overall vc_score
The overall vc_score is a weighted average reflecting the image's holistic visual complexity. Weight dimensions as follows:
- **High weight**: data_density, visual_encoding, domain_schema, cognitive_load (these drive VC the most)
- **Medium weight**: color_symbol, aesthetic_order
- **Low weight**: text_annotation (text volume alone is a weak VC signal)
The vc_score should be consistent with (but not necessarily the arithmetic mean of) the 7 dimension scores. A low text_annotation score should NOT substantially pull down the overall vc_score.

## Output Format
Return ONLY valid JSON (no markdown fences, no explanation outside JSON):
{
  "data_density": <float 0-1>,
  "visual_encoding": <float 0-1>,
  "text_annotation": <float 0-1>,
  "domain_schema": <float 0-1>,
  "color_symbol": <float 0-1>,
  "aesthetic_order": <float 0-1>,
  "cognitive_load": <float 0-1>,
  "vc_score": <float 0-1>,
  "data_density_explanation": "<1 sentence>",
  "visual_encoding_explanation": "<1 sentence>",
  "text_annotation_explanation": "<1 sentence>",
  "domain_schema_explanation": "<1 sentence>",
  "color_symbol_explanation": "<1 sentence>",
  "aesthetic_order_explanation": "<1 sentence>",
  "cognitive_load_explanation": "<1 sentence>",
  "summary": "<2-3 sentence overall description>"
}"""


# ── Helpers ─────────────────────────────────────────────────────────────────

def image_to_base64(path: Path) -> str:
    """Read local image file and return base64 string."""
    return base64.standard_b64encode(path.read_bytes()).decode('utf-8')


def image_url_to_base64(url: str) -> str:
    """Fetch image from URL and return base64 string."""
    import urllib.request
    with urllib.request.urlopen(url, timeout=30) as resp:
        return base64.standard_b64encode(resp.read()).decode('utf-8')


def get_media_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
            'gif': 'image/gif', 'webp': 'image/webp'}.get(ext.lstrip('.'), 'image/png')


def build_anchor_messages(source: str) -> list:
    """Build few-shot example messages from anchor images."""
    messages = []
    for anchor in ANCHORS:
        img_name = anchor['imageName']
        # Load anchor image
        b64 = _load_image(img_name, source)
        if b64 is None:
            print(f'  WARNING: Could not load anchor image {img_name}, skipping from few-shot')
            continue

        # User turn: just the image
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": get_media_type(img_name), "data": b64}
                },
                {
                    "type": "text",
                    "text": f"Score the visual complexity of this visualization image."
                }
            ]
        })

        # Assistant turn: the anchor scores as JSON
        response_obj = {}
        for dim in DIMENSIONS:
            response_obj[dim] = anchor['scores'][dim]
        response_obj['vc_score'] = anchor['vc_score']
        for dim in DIMENSIONS:
            response_obj[f'{dim}_explanation'] = anchor['explanations'][dim]
        response_obj['summary'] = anchor['summary']

        messages.append({
            "role": "assistant",
            "content": json.dumps(response_obj, indent=2)
        })

    # Mark the last anchor message for prompt caching — Anthropic caches
    # everything up to and including the block with cache_control, so all
    # 3 anchor image pairs are cached after the first API call.
    if messages:
        last_msg = messages[-1]
        # Convert string content to a content block so we can add cache_control
        if isinstance(last_msg['content'], str):
            last_msg['content'] = [
                {
                    "type": "text",
                    "text": last_msg['content'],
                    "cache_control": {"type": "ephemeral"}
                }
            ]

    return messages


def _load_image(img_name: str, source: str, url: str = None) -> str | None:
    """Load an image as base64 from local file or URL."""
    if source == 'local':
        path = LOCAL_IMG_DIR / img_name
        if path.exists():
            return image_to_base64(path)
        return None
    else:  # url
        if url:
            target = url
        else:
            target = f'https://raw.githubusercontent.com/c109363/ExperimentImage/main/AllDataResize/{img_name}'
        try:
            return image_url_to_base64(target)
        except Exception as e:
            print(f'  WARNING: Failed to fetch {img_name}: {e}')
            return None


def parse_response(raw: str) -> dict:
    """Parse Claude's JSON response, stripping markdown fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def load_existing_scores(scores_csv: Path) -> set:
    """Return set of filenames already in the scores CSV."""
    if not scores_csv.exists():
        return set()
    done = set()
    with open(scores_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row['filename'])
    return done


def append_scores_row(scores_csv: Path, filename: str, result: dict):
    """Append one row to the scores CSV."""
    write_header = not scores_csv.exists()
    with open(scores_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['filename'] + DIMENSIONS + ['vc_score'])
        row = [filename] + [result.get(d, '') for d in DIMENSIONS] + [result.get('vc_score', '')]
        writer.writerow(row)


def append_explanations_row(expl_csv: Path, filename: str, result: dict):
    """Append one row to the explanations CSV."""
    write_header = not expl_csv.exists()
    expl_cols = [f'{d}_explanation' for d in DIMENSIONS] + ['summary']
    with open(expl_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['filename'] + expl_cols)
        row = [filename] + [result.get(c, '') for c in expl_cols]
        writer.writerow(row)


# ── CSV write lock (for concurrent mode) ───────────────────────────────────
_csv_lock = threading.Lock()


def _score_one(client, img_name, img_url, source, anchor_messages, scores_csv, expl_csv):
    """Score a single image. Returns (img_name, vc_score|None, error|None)."""
    b64 = _load_image(img_name, source, url=img_url)
    if b64 is None:
        return (img_name, None, 'image not found')

    target_message = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": get_media_type(img_name), "data": b64}
            },
            {
                "type": "text",
                "text": "Score the visual complexity of this visualization image."
            }
        ]
    }
    messages = anchor_messages + [target_message]

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=messages
    )

    raw = response.content[0].text
    result = parse_response(raw)

    missing = [d for d in DIMENSIONS + ['vc_score'] if d not in result]
    if missing:
        print(f'  WARN [{img_name}]: missing keys {missing}; saving partial')

    with _csv_lock:
        append_scores_row(scores_csv, img_name, result)
        append_explanations_row(expl_csv, img_name, result)

    return (img_name, result.get('vc_score', '?'), None)


def _run_sequential(client, to_process, anchor_messages, source, scores_csv, expl_csv):
    """Original sequential processing."""
    ok, failed = 0, []
    for i, row in enumerate(to_process, 1):
        img_name = row['imageName']
        img_url  = row['imageURL']
        print(f'[{i}/{len(to_process)}] {img_name} ... ', end='', flush=True)
        try:
            name, vc, err = _score_one(client, img_name, img_url, source,
                                        anchor_messages, scores_csv, expl_csv)
            if err:
                print(f'SKIP ({err})')
                failed.append(img_name)
            else:
                print(f'OK  (vc_score={vc})')
                ok += 1
        except json.JSONDecodeError as e:
            print(f'JSON PARSE ERROR: {e}')
            failed.append(img_name)
        except anthropic.RateLimitError:
            print('RATE LIMITED — waiting 60s ...')
            time.sleep(60)
            failed.append(img_name)
        except Exception as e:
            print(f'ERROR: {e}')
            failed.append(img_name)
        if i < len(to_process):
            time.sleep(SLEEP_BETWEEN)
    return ok, failed


async def _score_one_async(aclient, sem, idx, total, img_name, img_url, source,
                           anchor_messages, scores_csv, expl_csv):
    """Score one image with semaphore-limited concurrency."""
    async with sem:
        b64 = _load_image(img_name, source, url=img_url)
        if b64 is None:
            print(f'[{idx}/{total}] {img_name} ... SKIP (image not found)')
            return (img_name, None)

        target_message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": get_media_type(img_name), "data": b64}
                },
                {"type": "text", "text": "Score the visual complexity of this visualization image."}
            ]
        }
        messages = anchor_messages + [target_message]

        try:
            response = await aclient.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=messages
            )
            raw = response.content[0].text
            result = parse_response(raw)

            missing = [d for d in DIMENSIONS + ['vc_score'] if d not in result]
            if missing:
                print(f'  WARN [{img_name}]: missing keys {missing}')

            with _csv_lock:
                append_scores_row(scores_csv, img_name, result)
                append_explanations_row(expl_csv, img_name, result)

            vc = result.get('vc_score', '?')
            print(f'[{idx}/{total}] {img_name} ... OK  (vc_score={vc})')
            return (img_name, vc)

        except anthropic.RateLimitError:
            print(f'[{idx}/{total}] {img_name} ... RATE LIMITED')
            await asyncio.sleep(60)
            return (img_name, None)
        except Exception as e:
            print(f'[{idx}/{total}] {img_name} ... ERROR: {e}')
            return (img_name, None)


async def _run_concurrent(api_key, to_process, anchor_messages, source,
                          scores_csv, expl_csv, concurrency):
    """Process images with bounded concurrency using async API."""
    aclient = anthropic.AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    total = len(to_process)
    print(f'Concurrent mode: {concurrency} workers\n')

    tasks = [
        _score_one_async(aclient, sem, i, total, row['imageName'], row['imageURL'],
                         source, anchor_messages, scores_csv, expl_csv)
        for i, row in enumerate(to_process, 1)
    ]

    results = await asyncio.gather(*tasks)
    ok = sum(1 for _, vc in results if vc is not None)
    failed = [name for name, vc in results if vc is None]
    return ok, failed


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Score visual complexity of images via Claude API')
    parser.add_argument('--limit', type=int, default=None, help='Process only first N images')
    parser.add_argument('--overwrite', action='store_true', help='Re-process already scored images')
    parser.add_argument('--source', choices=['local', 'url'], default='url',
                        help='Image source: local files or GitHub URLs (default: url)')
    parser.add_argument('--outdir', type=str, default=None, help='Output directory name')
    parser.add_argument('--images', type=str, nargs='*', default=None,
                        help='Specific image filenames to process')
    parser.add_argument('--input-csv', type=str, default=None,
                        help='CSV with imageName column (and optional imageURL) to use instead of default mapping')
    parser.add_argument('--concurrency', type=int, default=1,
                        help='Number of concurrent API calls (default: 1 = sequential)')
    args = parser.parse_args()

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print('ERROR: ANTHROPIC_API_KEY not set. Check your .env file.')
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # ── Load image list ─────────────────────────────────────────────────
    import pandas as pd
    if args.input_csv:
        input_df = pd.read_csv(args.input_csv)
        if 'imageURL' not in input_df.columns:
            # Join URLs from the master mapping
            mapping_df = pd.read_csv(MAPPING_CSV)
            url_map = dict(zip(mapping_df['imageName'], mapping_df['imageURL']))
            input_df['imageURL'] = input_df['imageName'].map(url_map).fillna('')
        all_images = input_df[['imageName', 'imageURL']].drop_duplicates('imageName').to_dict('records')
    else:
        mapping_df = pd.read_csv(MAPPING_CSV)
        all_images = mapping_df[['imageName', 'imageURL']].drop_duplicates('imageName').to_dict('records')

    if args.images:
        # Filter to specific images
        target_set = set(args.images)
        all_images = [r for r in all_images if r['imageName'] in target_set]

    if args.limit:
        all_images = all_images[:args.limit]

    # ── Output paths ────────────────────────────────────────────────────
    outdir = Path(args.outdir) if args.outdir else DEFAULT_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    scores_csv = outdir / 'vc_scores.csv'
    expl_csv   = outdir / 'vc_explanations.csv'

    # ── Skip already processed ──────────────────────────────────────────
    done = set() if args.overwrite else load_existing_scores(scores_csv)
    to_process = [r for r in all_images if r['imageName'] not in done]
    skip_count = len(all_images) - len(to_process)

    print(f'Model:          {MODEL}')
    print(f'Image source:   {args.source}')
    print(f'Output dir:     {outdir.resolve()}')
    print(f'Total images:   {len(all_images)}')
    print(f'Already done:   {skip_count}')
    print(f'To process:     {len(to_process)}')
    print()

    if not to_process:
        print('Nothing to process.')
        return

    # ── Build few-shot anchor messages (done once, cached in prompt) ────
    print('Loading anchor images for few-shot examples...')
    anchor_messages = build_anchor_messages(args.source)
    print(f'  {len(anchor_messages) // 2} anchor examples loaded.\n')

    # ── Process images ──────────────────────────────────────────────────
    concurrency = max(1, args.concurrency)

    if concurrency == 1:
        ok, failed = _run_sequential(client, to_process, anchor_messages, args.source,
                                      scores_csv, expl_csv)
    else:
        ok, failed = asyncio.run(
            _run_concurrent(api_key, to_process, anchor_messages, args.source,
                            scores_csv, expl_csv, concurrency)
        )

    print(f'\n{"="*50}')
    print(f'Done: {ok} scored, {len(failed)} failed')
    if failed:
        print(f'Failed images:')
        for f in failed:
            print(f'  {f}')
    print(f'Scores:       {scores_csv.resolve()}')
    print(f'Explanations: {expl_csv.resolve()}')


if __name__ == '__main__':
    main()
