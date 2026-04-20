"""
Claude API Visual Complexity (VC) Scoring Script — V4
=====================================================
Based on V3 with two additions:
  - Renames "dimensions" to "topics" using original full topic names
  - Adds topic relevance selection: up to 3 most relevant topics per image

Usage:
    python _vc_score_api_v4.py                     # process all images from mapping CSV
    python _vc_score_api_v4.py --limit 5           # process first N images
    python _vc_score_api_v4.py --overwrite         # re-process already scored images
    python _vc_score_api_v4.py --source local      # use local image files
    python _vc_score_api_v4.py --source url        # fetch images from URLs (default)
    python _vc_score_api_v4.py --outdir results    # custom output directory
"""

import os, sys, json, time, argparse, base64, re, csv, asyncio, threading
from pathlib import Path
from io import BytesIO
import anthropic
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

MODEL          = 'claude-opus-4-6'
MAX_TOKENS     = 2000
SLEEP_BETWEEN  = 0.5        # seconds between API calls

MAPPING_CSV    = Path(__file__).parent.parent / 'phrase_reduction_v2' / 'image_phrase_word_mapping.csv'
LOCAL_IMG_DIR  = Path(__file__).parent.parent / 'Claude_vc_prediction' / 'images'
DEFAULT_OUTDIR = Path(__file__).parent.parent / 'vc_api_scores_v4'

# Topic names (original full names from subtopic_taxonomy.csv)
TOPICS = [
    'Data Density / Image Clutter',
    'Visual Encoding Clarity',
    'Semantics / Text Legibility',
    'Schema',
    'Color, Symbol, and Texture Details',
    'Aesthetics Uncertainty',
    'Immediacy / Cognitive Load',
]

# Short keys for CSV columns (matching V3 dimension keys)
TOPIC_KEYS = [
    'data_density', 'visual_encoding', 'text_annotation',
    'domain_schema', 'color_symbol', 'aesthetic_order', 'cognitive_load'
]

# Map full topic name → short key
TOPIC_NAME_TO_KEY = dict(zip(TOPICS, TOPIC_KEYS))

# ── Anchor images (few-shot examples) ──────────────────────────────────────
ANCHORS = [
    {
        'imageName': 'VisC.503.6.png',
        'vc_score': 0.22,
        'scores': {
            'Data Density / Image Clutter': 0.15,
            'Visual Encoding Clarity': 0.10,
            'Semantics / Text Legibility': 0.20,
            'Schema': 0.15,
            'Color, Symbol, and Texture Details': 0.10,
            'Aesthetics Uncertainty': 0.10,
            'Immediacy / Cognitive Load': 0.20,
        },
        'explanations': {
            'Data Density / Image Clutter': 'Only 3 bars with error bars; very sparse data content.',
            'Visual Encoding Clarity': 'Standard vertical bars with simple hatching; minimal encoding variety.',
            'Semantics / Text Legibility': 'Basic axis labels and a title present but minimal text overall.',
            'Schema': 'Generic bar chart format; no specialized domain knowledge needed.',
            'Color, Symbol, and Texture Details': 'Monochrome black hatched bars on white; no color variety at all.',
            'Aesthetics Uncertainty': 'Clean, well-organized layout with ample whitespace.',
            'Immediacy / Cognitive Load': 'Immediately interpretable; very low effort to understand.',
        },
        'summary': 'A simple 3-bar chart with black hatching and error bars. Extremely low visual complexity across all dimensions.',
        'relevant_topics': [
            {'topic': 'Semantics / Text Legibility'},
            {'topic': 'Immediacy / Cognitive Load'},
        ],
        'topic_explanation': 'This minimal bar chart has almost no complexity drivers. The small amount of text and the trivially low cognitive demand are the only aspects worth noting.',
    },
    {
        'imageName': 'InfoVisJ.619.17.png',
        'vc_score': 0.54,
        'scores': {
            'Data Density / Image Clutter': 0.55,
            'Visual Encoding Clarity': 0.50,
            'Semantics / Text Legibility': 0.45,
            'Schema': 0.60,
            'Color, Symbol, and Texture Details': 0.50,
            'Aesthetics Uncertainty': 0.45,
            'Immediacy / Cognitive Load': 0.55,
        },
        'explanations': {
            'Data Density / Image Clutter': 'Moderate number of data points in a 2D scatterplot with some grouping structure visible.',
            'Visual Encoding Clarity': 'Points plotted in a biplot layout with directional arrows; moderate encoding complexity.',
            'Semantics / Text Legibility': 'Axis labels and some point labels present; moderate text volume.',
            'Schema': 'Biplot/PCA projection requires some statistical familiarity.',
            'Color, Symbol, and Texture Details': 'A few colors distinguish groups; moderate palette diversity.',
            'Aesthetics Uncertainty': 'Reasonably organized but arrows and overlapping labels add mild clutter.',
            'Immediacy / Cognitive Load': 'Requires understanding of biplots and PCA; moderate interpretation effort.',
        },
        'summary': 'A PCA biplot with directional arrows and grouped points. Moderate complexity requiring some statistical knowledge.',
        'relevant_topics': [
            {'topic': 'Schema'},
            {'topic': 'Immediacy / Cognitive Load'},
            {'topic': 'Data Density / Image Clutter'},
        ],
        'topic_explanation': 'The biplot/PCA schema drives most of the complexity, requiring statistical knowledge. The cognitive load of interpreting directional arrows and groupings is substantial, and the moderate data density adds to the visual demand.',
    },
    {
        'imageName': 'InfoVisJ.1149.6(1).png',
        'vc_score': 0.95,
        'scores': {
            'Data Density / Image Clutter': 0.90,
            'Visual Encoding Clarity': 0.85,
            'Semantics / Text Legibility': 0.80,
            'Schema': 0.95,
            'Color, Symbol, and Texture Details': 0.70,
            'Aesthetics Uncertainty': 0.85,
            'Immediacy / Cognitive Load': 0.95,
        },
        'explanations': {
            'Data Density / Image Clutter': 'Multiple coordinated panels (7 views) each showing dense text and data; extremely high information volume.',
            'Visual Encoding Clarity': 'Mixed encoding types across panels: lists, highlighted text, bar-like indicators, tag clouds; high variety.',
            'Semantics / Text Legibility': 'Extensive text content across all panels; the visualization IS largely text-based with highlights and annotations.',
            'Schema': 'Specialized text analysis/collation tool; requires understanding of multi-view coordinated systems.',
            'Color, Symbol, and Texture Details': 'Multiple highlight colors for text segments; moderate color use with functional meaning.',
            'Aesthetics Uncertainty': 'Complex multi-panel layout with varied panel sizes; visually dense and somewhat overwhelming.',
            'Immediacy / Cognitive Load': 'Must track relationships across 7 coordinated views with dense text; very high cognitive demand.',
        },
        'summary': 'A multi-panel text collation tool with 7 coordinated views showing highlighted text passages, bar indicators, and tag clouds. Extremely high visual complexity requiring sustained multi-view comparison.',
        'relevant_topics': [
            {'topic': 'Immediacy / Cognitive Load'},
            {'topic': 'Schema'},
            {'topic': 'Data Density / Image Clutter'},
        ],
        'topic_explanation': 'The extreme cognitive load of tracking 7 coordinated views is the primary complexity driver. The specialized multi-view schema demands domain expertise, and the sheer density of information across all panels compounds the difficulty.',
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

## Scoring Topics (each 0–1)

1. **Data Density / Image Clutter** — The perceived amount, richness, or depth of data content. Considers information volume, element quantity, and visual clutter/overlap.
2. **Visual Encoding Clarity** — The variety, type, and complexity of graphical forms (shapes, lines, marks) and how spatial layout, scale, and encoding interpretability contribute to complexity.
3. **Semantics / Text Legibility** — The sheer quantity and density of text elements (titles, axis labels, legends, captions, annotations, in-chart labels). Score based on volume only: more text = higher score, regardless of legibility.
4. **Schema** — Whether specialized domain knowledge is needed, including dimensionality (2D/3D), structural complexity, and abstraction level.
5. **Color, Symbol, and Texture Details** — Range, variety, and arrangement of colors, plus use of symbols, textures, and non-color graphical markers.
6. **Aesthetics Uncertainty** — How visually cluttered, dense, or disordered the layout appears. Higher = more cluttered/overwhelming. A clean minimal layout scores low; a crowded layout with overlapping elements scores high.
7. **Immediacy / Cognitive Load** — Overall ease or difficulty of interpreting the visualization. Considers interpretive difficulty, semantic clarity, and processing time/effort.

## Overall vc_score
The overall vc_score is a weighted average reflecting the image's holistic visual complexity. Weight topics as follows:
- **High weight**: Data Density / Image Clutter, Visual Encoding Clarity, Schema, Immediacy / Cognitive Load (these drive VC the most)
- **Medium weight**: Color, Symbol, and Texture Details; Aesthetics Uncertainty
- **Low weight**: Semantics / Text Legibility (text volume alone is a weak VC signal)
The vc_score should be consistent with (but not necessarily the arithmetic mean of) the 7 topic scores. A low Semantics / Text Legibility score should NOT substantially pull down the overall vc_score.

## Topic Relevance
After scoring, select up to **3 topics** (by name) that are **most relevant** to the visual complexity of this image. List them in order of relevance, most relevant first. Then write briefly why these topics are most relevant.

## Output Format
Return ONLY valid JSON (no markdown fences, no explanation outside JSON):
{
  "Data Density / Image Clutter": <float 0-1>,
  "Visual Encoding Clarity": <float 0-1>,
  "Semantics / Text Legibility": <float 0-1>,
  "Schema": <float 0-1>,
  "Color, Symbol, and Texture Details": <float 0-1>,
  "Aesthetics Uncertainty": <float 0-1>,
  "Immediacy / Cognitive Load": <float 0-1>,
  "vc_score": <float 0-1>,
  "Data Density / Image Clutter_explanation": "<1 sentence>",
  "Visual Encoding Clarity_explanation": "<1 sentence>",
  "Semantics / Text Legibility_explanation": "<1 sentence>",
  "Schema_explanation": "<1 sentence>",
  "Color, Symbol, and Texture Details_explanation": "<1 sentence>",
  "Aesthetics Uncertainty_explanation": "<1 sentence>",
  "Immediacy / Cognitive Load_explanation": "<1 sentence>",
  "summary": "<2-3 sentence overall description>",
  "topics": [
    {"topic": "<topic_name>"}
  ],
  "topic_explanation": "<1-3 concise sentences (max 100 words total) explaining why these topics are relevant>"
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
        b64 = _load_image(img_name, source)
        if b64 is None:
            print(f'  WARNING: Could not load anchor image {img_name}, skipping from few-shot')
            continue

        # User turn
        messages.append({
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
        })

        # Assistant turn: anchor scores as JSON using full topic names
        response_obj = {}
        for topic in TOPICS:
            response_obj[topic] = anchor['scores'][topic]
        response_obj['vc_score'] = anchor['vc_score']
        for topic in TOPICS:
            response_obj[f'{topic}_explanation'] = anchor['explanations'][topic]
        response_obj['summary'] = anchor['summary']
        response_obj['topics'] = anchor['relevant_topics']
        response_obj['topic_explanation'] = anchor['topic_explanation']

        messages.append({
            "role": "assistant",
            "content": json.dumps(response_obj, indent=2)
        })

    # Mark the last anchor message for prompt caching
    if messages:
        last_msg = messages[-1]
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
    else:
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


def _normalize_result(result: dict) -> dict:
    """Map full topic names to short keys for CSV output and extract topic relevance."""
    out = {}
    for topic, key in TOPIC_NAME_TO_KEY.items():
        out[key] = result.get(topic, '')
        out[f'{key}_explanation'] = result.get(f'{topic}_explanation', '')
    out['vc_score'] = result.get('vc_score', '')
    out['summary'] = result.get('summary', '')

    # Topic relevance
    topics_list = result.get('topics', [])
    topic_names = [t['topic'] for t in topics_list if isinstance(t, dict) and 'topic' in t]
    out['relevant_topics'] = '; '.join(topic_names)
    out['topic_explanation'] = result.get('topic_explanation', '')
    return out


def append_scores_row(scores_csv: Path, filename: str, result: dict):
    """Append one row to the scores CSV."""
    write_header = not scores_csv.exists()
    with open(scores_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['filename'] + TOPIC_KEYS + ['vc_score', 'relevant_topics'])
        row = [filename] + [result.get(k, '') for k in TOPIC_KEYS] + [
            result.get('vc_score', ''), result.get('relevant_topics', '')]
        writer.writerow(row)


def append_explanations_row(expl_csv: Path, filename: str, result: dict):
    """Append one row to the explanations CSV."""
    write_header = not expl_csv.exists()
    expl_cols = [f'{k}_explanation' for k in TOPIC_KEYS] + ['summary', 'topic_explanation']
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
    result = _normalize_result(result)

    missing = [k for k in TOPIC_KEYS + ['vc_score'] if k not in result or result[k] == '']
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
            result = _normalize_result(result)

            missing = [k for k in TOPIC_KEYS + ['vc_score'] if k not in result or result[k] == '']
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
    parser = argparse.ArgumentParser(description='Score visual complexity of images via Claude API (V4)')
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
    parser.add_argument('--summary', action='store_true',
                        help='Generate topic_relevance_summary.csv from existing results and exit')
    args = parser.parse_args()

    api_key = os.environ.get('ANTHROPIC_API_KEY')

    # ── Summary mode ────────────────────────────────────────────────────
    if args.summary:
        import pandas as pd
        outdir = Path(args.outdir) if args.outdir else DEFAULT_OUTDIR
        scores_csv = outdir / 'vc_scores.csv'
        expl_csv   = outdir / 'vc_explanations.csv'
        if not scores_csv.exists() or not expl_csv.exists():
            print(f'ERROR: Missing output files in {outdir.resolve()}')
            sys.exit(1)
        scores = pd.read_csv(scores_csv)[['filename', 'relevant_topics']]
        expl   = pd.read_csv(expl_csv)[['filename', 'topic_explanation']]
        merged = (scores.merge(expl, on='filename')
                  .rename(columns={'filename': 'imageName'}))
        # If --input-csv given, order rows to match its imageName column
        if args.input_csv:
            order = pd.read_csv(args.input_csv)[['imageName']]
            merged = order.merge(merged, on='imageName', how='left')
        else:
            merged = merged.sort_values('imageName').reset_index(drop=True)
        out_path = outdir / 'topic_relevance_summary.csv'
        merged.to_csv(out_path, index=False)
        print(f'Saved {len(merged)} rows to {out_path.resolve()}')
        print(merged.to_string(index=False))
        return

    if not api_key:
        print('ERROR: ANTHROPIC_API_KEY not set. Check your .env file.')
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # ── Load image list ─────────────────────────────────────────────────
    import pandas as pd
    if args.input_csv:
        input_df = pd.read_csv(args.input_csv)
        if 'imageURL' not in input_df.columns:
            mapping_df = pd.read_csv(MAPPING_CSV)
            url_map = dict(zip(mapping_df['imageName'], mapping_df['imageURL']))
            input_df['imageURL'] = input_df['imageName'].map(url_map).fillna('')
        all_images = input_df[['imageName', 'imageURL']].drop_duplicates('imageName').to_dict('records')
    else:
        mapping_df = pd.read_csv(MAPPING_CSV)
        all_images = mapping_df[['imageName', 'imageURL']].drop_duplicates('imageName').to_dict('records')

    if args.images:
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

    # ── Build few-shot anchor messages ──────────────────────────────────
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
