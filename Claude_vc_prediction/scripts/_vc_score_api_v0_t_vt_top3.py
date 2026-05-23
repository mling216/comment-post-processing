"""
Claude API Visual Complexity (VC) Scoring — V0+Topic+VisTypeDefs (Top-3 variant)
=================================================================================
V0+T+VT: V0+Topic prompt extended with the 9 vis-type definitions so the LLM
can identify the visualization type and apply type-appropriate complexity
reasoning.  Also includes the top-3 topic-selection task.

Usage:
    python _vc_score_api_v0_t_vt_top3.py --input-csv ../Claude_vc_prediction/gt_all_66.csv --outdir ../results/vc_api_topicsel_v0_t_vt --concurrency 5 --model claude-opus-4-6
"""

import os, sys, json, time, argparse, base64, csv, asyncio, threading
from pathlib import Path
import anthropic
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

MODEL          = 'claude-sonnet-4-6'
MAX_TOKENS     = 1024
THINKING_BUDGET = 4000
USE_THINKING   = False
SLEEP_BETWEEN  = 0.5

MAPPING_CSV    = Path(__file__).parent.parent / 'phrase_reduction_v2' / 'image_phrase_word_mapping.csv'
DEFAULT_OUTDIR = Path(__file__).parent.parent / 'results' / 'vc_api_topicsel_v0_t_vt'

DIM_KEYS = ['data_density', 'visual_encoding', 'text_annotation',
            'domain_schema', 'color_symbol', 'aesthetic_order', 'cognitive_load']

# ── V0+Topic+VisTypeDefs System Prompt ──────────────────────────────────────
SYSTEM_PROMPT = """You are a visual complexity (VC) scoring expert for data visualizations.

**Visual Complexity** measures the amount of detail, intricacy, and perceptual/cognitive demand of a visualization image. It is scored on a 0–1 scale where 0 = minimal complexity and 1 = maximum complexity.

You will receive a single visualization image with NO accompanying text or comments. Score it purely from what you see.

## Visualization Type Definitions

Use the following definitions to identify the type of visualization, which will help you apply type-appropriate complexity reasoning.

- **Bar** (Generalized Bar Representations): Graphs with straight bars arranged on a straight or curved baseline; bar heights/lengths are proportional to values. Includes bar charts, stacked bar charts, box plots, sunburst diagrams.
- **Point** (Point-based Representations): Representations using point locations shown as dots, circles, or other shapes (spheres, triangles, stars). Includes scatterplots, point clouds, dot plots, bubble charts.
- **Line** (Line-based Representations): Representations emphasizing straight or curved lines encoding magnitude or trends. Includes line charts, parallel coordinates, contour lines, radar/spider charts, streamlines.
- **Node-link** (Node-link Trees/Graphs, Networks, Meshes): Representations using points with explicit connections conveying relationships. Includes node-link diagrams, trees, graphs, meshes, arc diagrams, Sankey diagrams.
- **Area** (Area-based Representations): Representations focused on 2D areas or surfaces, including geographic regions or polygons. Includes area charts, streamgraphs, ThemeRiver, violin plots, cartograms, histograms, treemaps, pie charts.
- **Grid** (Generalized Matrix/Grid): Representations with discrete spatial grid structures (rectangular, hexagonal, or cubic cells) that may contain glyphs or color encodings. Includes network matrices, discrete heatmaps, scarf/strip plots, space-time cubes.
- **Cont.-ColorPatn** (Continuous Color/Grey-scale and Textures): Representations of structured patterns across an image or 3D object via changes in intensity, hue, brightness, or saturation (typically smooth/continuous). Includes LIC, Spot Noise, ISA flow fields, continuous heatmaps, intensity fields.
- **Glyph** (Glyph-based Representations): Multiple small independent visual representations encoding multiple data attributes, typically placed in a meaningful spatial arrangement for comparison. Includes star glyphs, 3D glyphs, Chernoff faces, vector field glyphs.
- **Text** (Text-based Representations): Representations using properties of letters/words (font size, color, width, style) to encode data. Includes tag clouds, word trees, parallel tag clouds, typomaps.

## Topics to Consider When Scoring

When assessing visual complexity, consider the following 7 topics that commonly drive complexity perception. You do not need to score each topic individually — use them as mental checkpoints to arrive at a holistic vc_score.

1. **Data Density / Image Clutter** — The perceived amount, richness, or depth of data content. Considers information volume, element quantity, and visual clutter/overlap.
2. **Visual Encoding Clarity** — The variety, type, and complexity of graphical forms (shapes, lines, marks) and how spatial layout, scale, and encoding interpretability contribute to complexity.
3. **Semantics / Text Legibility** — The quantity and density of text elements (titles, axis labels, legends, captions, annotations, in-chart labels).
4. **Schema** — Whether specialized domain knowledge is needed, including dimensionality (2D/3D), structural complexity, and abstraction level.
5. **Color, Symbol, and Texture Details** — Range, variety, and arrangement of colors, plus use of symbols, textures, and non-color graphical markers.
6. **Aesthetics Uncertainty** — How visually cluttered, dense, or disordered the layout appears. Higher = more cluttered/overwhelming. A clean minimal layout scores low; a crowded layout with overlapping elements scores high.
7. **Immediacy / Cognitive Load** — Overall ease or difficulty of interpreting the visualization. Considers interpretive difficulty, semantic clarity, and processing time/effort.

## Topic-Key Mapping
Each topic corresponds to a short key:
- "Data Density / Image Clutter" → `data_density`
- "Visual Encoding Clarity" → `visual_encoding`
- "Semantics / Text Legibility" → `text_annotation`
- "Schema" → `domain_schema`
- "Color, Symbol, and Texture Details" → `color_symbol`
- "Aesthetics Uncertainty" → `aesthetic_order`
- "Immediacy / Cognitive Load" → `cognitive_load`

## Output Format
Return ONLY valid JSON (no markdown fences, no explanation outside JSON):
{
  "vc_score": <float 0-1>,
  "top3_topics": ["<key>", "<key>", "<key>"],
  "explanation": "<2-3 sentence justification referencing the topics that most influenced the score>"
}

Where `top3_topics` is an ordered list (most → least influential) of exactly three distinct keys drawn from: data_density, visual_encoding, text_annotation, domain_schema, color_symbol, aesthetic_order, cognitive_load."""


# ── Helpers ─────────────────────────────────────────────────────────────────

def image_url_to_base64(url: str) -> str:
    import urllib.request
    with urllib.request.urlopen(url, timeout=30) as resp:
        return base64.standard_b64encode(resp.read()).decode('utf-8')


def get_media_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
            'gif': 'image/gif', 'webp': 'image/webp'}.get(ext.lstrip('.'), 'image/png')


def _load_image(img_name: str, url: str = None) -> str | None:
    target = url or f'https://raw.githubusercontent.com/c109363/ExperimentImage/main/AllDataResize/{img_name}'
    try:
        return image_url_to_base64(target)
    except Exception as e:
        print(f'  WARNING: Failed to fetch {img_name}: {e}')
        return None


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


def _normalize_top3(val) -> str:
    if not isinstance(val, list):
        return ''
    keys = [str(k).strip() for k in val if str(k).strip() in DIM_KEYS]
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k); out.append(k)
    return ';'.join(out[:3])


def append_scores_row(scores_csv: Path, filename: str, result: dict):
    write_header = not scores_csv.exists()
    with open(scores_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['filename', 'vc_score', 'top3_topics'])
        writer.writerow([filename, result.get('vc_score', ''),
                         _normalize_top3(result.get('top3_topics'))])


def append_explanations_row(expl_csv: Path, filename: str, result: dict):
    write_header = not expl_csv.exists()
    with open(expl_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['filename', 'explanation'])
        writer.writerow([filename, result.get('explanation', '')])


_csv_lock = threading.Lock()


# ── Async scoring ────────────────────────────────────────────────────────────

async def _score_one_async(aclient, sem, idx, total, img_name, img_url,
                           scores_csv, expl_csv):
    async with sem:
        b64 = _load_image(img_name, url=img_url)
        if b64 is None:
            print(f'[{idx}/{total}] {img_name} ... SKIP (image not found)')
            return (img_name, None)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": get_media_type(img_name), "data": b64}
                    },
                    {"type": "text", "text": "Score the visual complexity of this visualization image."}
                ]
            }
        ]

        try:
            api_kwargs = dict(model=MODEL, messages=messages,
                              max_tokens=MAX_TOKENS + THINKING_BUDGET if USE_THINKING else MAX_TOKENS,
                              system=[{"type": "text", "text": SYSTEM_PROMPT}])
            if USE_THINKING:
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


async def _run_concurrent(api_key, to_process, scores_csv, expl_csv, concurrency):
    aclient = anthropic.AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    total = len(to_process)
    print(f'Concurrent mode: {concurrency} workers\n')

    tasks = [
        _score_one_async(aclient, sem, i, total, row['imageName'], row['imageURL'],
                         scores_csv, expl_csv)
        for i, row in enumerate(to_process, 1)
    ]

    results = await asyncio.gather(*tasks)
    ok = sum(1 for _, vc in results if vc is not None)
    failed = [name for name, vc in results if vc is None]
    return ok, failed


def _run_sequential(client, to_process, scores_csv, expl_csv):
    ok, failed = 0, []
    for i, row in enumerate(to_process, 1):
        img_name = row['imageName']
        img_url  = row['imageURL']
        print(f'[{i}/{len(to_process)}] {img_name} ... ', end='', flush=True)
        try:
            b64 = _load_image(img_name, url=img_url)
            if b64 is None:
                print('SKIP (image not found)')
                failed.append(img_name)
                continue

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": get_media_type(img_name), "data": b64}
                        },
                        {"type": "text", "text": "Score the visual complexity of this visualization image."}
                    ]
                }
            ]

            api_kwargs = dict(model=MODEL, messages=messages,
                              max_tokens=MAX_TOKENS + THINKING_BUDGET if USE_THINKING else MAX_TOKENS,
                              system=[{"type": "text", "text": SYSTEM_PROMPT}])
            if USE_THINKING:
                api_kwargs['thinking'] = {"type": "adaptive"}
            else:
                api_kwargs['temperature'] = 0
            response = client.messages.create(**api_kwargs)
            raw = next(b.text for b in response.content if b.type == 'text')
            result = parse_response(raw)

            append_scores_row(scores_csv, img_name, result)
            append_explanations_row(expl_csv, img_name, result)

            print(f'OK  (vc_score={result.get("vc_score", "?")})')
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='V0+Topic+VisTypeDefs VC Scoring via Claude API')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--input-csv', type=str, default=None,
                        help='CSV with imageName column')
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--thinking', action='store_true',
                        help='Enable extended thinking (adaptive)')
    parser.add_argument('--model', type=str, default=None,
                        help='Override model (e.g. claude-opus-4-6 or claude-sonnet-4-6)')
    args = parser.parse_args()
    global USE_THINKING, MODEL
    USE_THINKING = args.thinking
    if args.model:
        MODEL = args.model

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print('ERROR: ANTHROPIC_API_KEY not set. Check your .env file.')
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

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

    if args.overwrite:
        scores_csv.unlink(missing_ok=True)
        expl_csv.unlink(missing_ok=True)

    already_done = load_existing_scores(scores_csv)
    to_process = [r for r in all_images if r['imageName'] not in already_done]

    print(f'Model: {MODEL}  |  Thinking: {USE_THINKING}')
    print(f'Total images : {len(all_images)}')
    print(f'Already done : {len(already_done)}')
    print(f'To process   : {len(to_process)}')
    print(f'Output dir   : {outdir}\n')

    if not to_process:
        print('Nothing to do — all images already scored.')
        return

    if args.concurrency > 1:
        ok, failed = asyncio.run(
            _run_concurrent(api_key, to_process, scores_csv, expl_csv, args.concurrency))
    else:
        ok, failed = _run_sequential(client, to_process, scores_csv, expl_csv)

    print(f'\nDone. OK={ok}  Failed={len(failed)}')
    if failed:
        print('Failed images:', failed)


if __name__ == '__main__':
    main()
