"""
Claude API Visual Complexity (VC) Scoring — V0+TW + V1-Detailed-Method (V0+TWdet)
==================================================================================
Base: V0+TW (topics + weighted, no calibration).
Added from V1 (v3): (a) per-dimension anchor supervision and (b) required
per-dimension numeric output with explanations.

This tests whether V1's gain over V0+TWCA comes from the detailed-method
layer alone (per-dim anchors + per-dim output), independent of the
calibration block that V1 also carries.

Usage:
    python _vc_score_api_v0_tw_det.py --input-csv ../Claude_vc_prediction/gt_all_46.csv --outdir ../results/vc_api_46gt_v0_tw_det --concurrency 5 --model claude-opus-4-6
    python _vc_score_api_v0_tw_det.py --input-csv ../Claude_vc_prediction/gt_all_46.csv --outdir ../results/vc_api_46gt_v0_tw_det_r2 --concurrency 5 --model claude-opus-4-6
"""

import os, sys, json, time, argparse, base64, csv, asyncio, threading
from pathlib import Path
import anthropic
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

MODEL          = 'claude-sonnet-4-6'
MAX_TOKENS     = 1500
SLEEP_BETWEEN  = 0.5

MAPPING_CSV    = Path(__file__).parent.parent / 'phrase_reduction_v2' / 'image_phrase_word_mapping.csv'
LOCAL_IMG_DIR  = Path(__file__).parent.parent / 'Claude_vc_prediction' / 'images'
DEFAULT_OUTDIR = Path(__file__).parent.parent / 'results' / 'vc_api_46gt_v0_tw_det'

DIMENSIONS = [
    'data_density', 'visual_encoding', 'text_annotation',
    'domain_schema', 'color_symbol', 'aesthetic_order', 'cognitive_load'
]

# ── Anchors with per-dimension supervision (copied from V1/v3) ─────────────
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

# ── System Prompt — V0 + Topics + Weighted + Per-Dimension Output (no calibration) ──
SYSTEM_PROMPT = """You are a visual complexity (VC) scoring expert for data visualizations.

**Visual Complexity** measures the amount of detail, intricacy, and perceptual/cognitive demand of a visualization image. It is scored on a 0–1 scale where 0 = minimal complexity and 1 = maximum complexity.

You will receive a single visualization image with NO accompanying text or comments. Score it purely from what you see.

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
    return base64.standard_b64encode(path.read_bytes()).decode('utf-8')


def image_url_to_base64(url: str) -> str:
    import urllib.request
    with urllib.request.urlopen(url, timeout=30) as resp:
        return base64.standard_b64encode(resp.read()).decode('utf-8')


def get_media_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
            'gif': 'image/gif', 'webp': 'image/webp'}.get(ext.lstrip('.'), 'image/png')


def _load_image(img_name: str, source: str, url: str = None) -> str | None:
    if source == 'local':
        path = LOCAL_IMG_DIR / img_name
        if path.exists():
            return image_to_base64(path)
        return None
    else:
        target = url or f'https://raw.githubusercontent.com/c109363/ExperimentImage/main/AllDataResize/{img_name}'
        try:
            return image_url_to_base64(target)
        except Exception as e:
            print(f'  WARNING: Failed to fetch {img_name}: {e}')
            return None


def build_anchor_messages(source: str) -> list:
    messages = []
    for anchor in ANCHORS:
        img_name = anchor['imageName']
        b64 = _load_image(img_name, source)
        if b64 is None:
            print(f'  WARNING: Could not load anchor image {img_name}')
            continue

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": get_media_type(img_name), "data": b64}
                },
                {"type": "text", "text": "Score the visual complexity of this visualization image."}
            ]
        })

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


def parse_response(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def load_existing_scores(scores_csv: Path) -> set:
    if not scores_csv.exists():
        return set()
    done = set()
    with open(scores_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row['filename'])
    return done


def append_scores_row(scores_csv: Path, filename: str, result: dict):
    write_header = not scores_csv.exists()
    with open(scores_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['filename'] + DIMENSIONS + ['vc_score'])
        row = [filename] + [result.get(d, '') for d in DIMENSIONS] + [result.get('vc_score', '')]
        writer.writerow(row)


def append_explanations_row(expl_csv: Path, filename: str, result: dict):
    write_header = not expl_csv.exists()
    expl_cols = [f'{d}_explanation' for d in DIMENSIONS] + ['summary']
    with open(expl_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['filename'] + expl_cols)
        row = [filename] + [result.get(c, '') for c in expl_cols]
        writer.writerow(row)


_csv_lock = threading.Lock()


async def _score_one_async(aclient, sem, idx, total, img_name, img_url, source,
                           anchor_messages, scores_csv, expl_csv, use_thinking):
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
            api_kwargs = dict(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=messages,
            )
            if use_thinking:
                api_kwargs['thinking'] = {"type": "adaptive"}
            else:
                api_kwargs['temperature'] = 0
            response = await aclient.messages.create(**api_kwargs)
            raw = next(b.text for b in response.content if b.type == 'text')
            result = parse_response(raw)

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
                          scores_csv, expl_csv, concurrency, use_thinking):
    aclient = anthropic.AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    total = len(to_process)
    print(f'Concurrent mode: {concurrency} workers\n')

    tasks = [
        _score_one_async(aclient, sem, i, total, row['imageName'], row['imageURL'],
                         source, anchor_messages, scores_csv, expl_csv, use_thinking)
        for i, row in enumerate(to_process, 1)
    ]

    results = await asyncio.gather(*tasks)
    ok = sum(1 for _, vc in results if vc is not None)
    failed = [name for name, vc in results if vc is None]
    return ok, failed


def main():
    parser = argparse.ArgumentParser(description='V0+TW + V1-detailed-method VC Scoring via Claude API')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--source', choices=['local', 'url'], default='url')
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--input-csv', type=str, default=None)
    parser.add_argument('--concurrency', type=int, default=5)
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()
    global MODEL
    if args.model:
        MODEL = args.model

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print('ERROR: ANTHROPIC_API_KEY not set.')
        sys.exit(1)

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

    if args.limit:
        all_images = all_images[:args.limit]

    outdir = Path(args.outdir) if args.outdir else DEFAULT_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    scores_csv = outdir / 'vc_scores.csv'
    expl_csv   = outdir / 'vc_explanations.csv'

    done = set() if args.overwrite else load_existing_scores(scores_csv)
    to_process = [r for r in all_images if r['imageName'] not in done]
    skip_count = len(all_images) - len(to_process)

    print(f'=== V0+TW + V1-detailed-method (topics + weighted + per-dim anchors + per-dim output, NO calibration) ===')
    print(f'Model:          {MODEL}')
    print(f'Thinking:       {args.thinking}')
    print(f'Output dir:     {outdir.resolve()}')
    print(f'Total images:   {len(all_images)}')
    print(f'Already done:   {skip_count}')
    print(f'To process:     {len(to_process)}')
    print()

    if not to_process:
        print('Nothing to process.')
        return

    print('Loading anchor images with per-dimension supervision ...')
    anchor_messages = build_anchor_messages(args.source)
    print(f'  {len(anchor_messages) // 2} anchors loaded.\n')

    concurrency = max(1, args.concurrency)
    ok, failed = asyncio.run(_run_concurrent(
        api_key, to_process, anchor_messages, args.source,
        scores_csv, expl_csv, concurrency, args.thinking))

    print(f'\n{"="*50}')
    print(f'Done: {ok} scored, {len(failed)} failed')
    if failed:
        print('Failed images:')
        for f in failed:
            print(f'  {f}')


if __name__ == '__main__':
    main()
