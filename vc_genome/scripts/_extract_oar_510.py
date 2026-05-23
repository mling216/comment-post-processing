"""
O/A/R extraction for all 510 images (9 main VisTypes) — B and V1 conditions.
=============================================================================
Extends the existing 63/66-image eval set extractions to cover the full
510-image dataset used in the paper's main analysis.

Conditions:
  B  — topics + sentiment-tagged participant phrases (text only, no image)
  V1 — topics + rendered image (vision only, no exemplars)

Outputs (appended to / merged with existing files):
  vc_genome_output_full/three_conditions/oar_B_510.json
  vc_genome_output_full/three_conditions/oar_V1_510.json

Existing oar_B.json / oar_V1.json (63/66 eval images) are used as a warm
start — already-done images are skipped automatically.

Usage:
  python scripts/_extract_oar_510.py --condition B
  python scripts/_extract_oar_510.py --condition V1
  python scripts/_extract_oar_510.py --condition all   # B then V1
  python scripts/_extract_oar_510.py --condition all --limit 10  # smoke test
"""
from __future__ import annotations
import os, sys, json, asyncio, argparse, base64
from pathlib import Path

import aiohttp
import pandas as pd
from dotenv import load_dotenv
import anthropic

ROOT = Path(__file__).parent.parent.parent
load_dotenv(dotenv_path=ROOT.parent / '.env')

MODEL       = 'claude-sonnet-4-6'
MAX_TOKENS  = 2048
TEMPERATURE = 0.0
CONCURRENCY = 5

IMAGE_BASE_URL = 'https://raw.githubusercontent.com/c109363/ExperimentImage/main/AllDataResize/'

DATA_CSV = ROOT / 'phrase_reduction_v2' / 'image_compiled_phrases.csv'
TC_DIR   = ROOT / 'vc_genome_output_full' / 'three_conditions'
TC_DIR.mkdir(parents=True, exist_ok=True)

# The 9 main VisTypes that make up the 510-image corpus
NINE_VISTYPES = ['Area', 'Bar', 'Cont.-ColorPatn', 'Glyph', 'Grid',
                 'Line', 'Node-link', 'Point', 'Text']

# ── 7 Topics block (shared with _extract_three_conditions.py) ─────────────
TOPICS_BLOCK = """## Topics (the 7-topic taxonomy)

Use these 7 topics when labeling attributes and relationships. Each attribute
and relationship must be tagged with exactly one topic.

1. **Data Density / Image Clutter** — The perceived amount, richness, or depth of data content. Considers information volume, element quantity, and visual clutter/overlap.
2. **Visual Encoding Clarity** — The variety, type, and complexity of graphical forms (shapes, lines, marks) and how spatial layout, scale, and encoding interpretability contribute to complexity.
3. **Semantics / Text Legibility** — The quantity and density of text elements (titles, axis labels, legends, captions, annotations, in-chart labels).
4. **Schema** — Whether specialized domain knowledge is needed, including dimensionality (2D/3D), structural complexity, and abstraction level.
5. **Color, Symbol, and Texture Details** — Range, variety, and arrangement of colors, plus use of symbols, textures, and non-color graphical markers.
6. **Aesthetics Uncertainty** — How visually cluttered, dense, or disordered the layout appears. Higher = more cluttered/overwhelming.
7. **Immediacy / Cognitive Load** — Overall ease or difficulty of interpreting the visualization. Considers interpretive difficulty, semantic clarity, and processing time/effort."""

OUTPUT_SCHEMA_BLOCK = """## Output Schema

Return ONLY valid JSON (no markdown fences, no prose outside JSON):
{
  "objects": [
    {"id": 1, "name": "object_name", "region": "data_area|axes|legend|title|annotation|overall"}
  ],
  "attributes": [
    {"object_id": 1, "attr": "short_snake_case_phrase", "sentiment": "+" or "-", "topic": "<one of the 7 topics>"}
  ],
  "relationships": [
    {"subj": 1, "pred": "snake_case_predicate", "obj": 2, "sentiment": "+" or "-", "topic": "<one of the 7 topics>"}
  ]
}

Guidelines:
- Extract elements proportional to the richness of the input. Typical ranges: 2–6 objects, 3–8 attributes, 1–5 relationships; sparse inputs may produce fewer and rich inputs may exceed these.
- Each list must contain at least one entry.
- Object names: single lowercase word or snake_case.
- Attribute text: snake_case, max 4 words.
- Sentiment: '+' increases perceived complexity, '-' decreases it.
- Topic must be exactly one of the seven topic titles listed above."""

SYSTEM_B = f"""You are a visual complexity annotation expert. You will receive, for a single data-visualization image:
  (a) a list of original complexity phrases extracted from real participant comments about the image, each tagged with a sentiment marker `(+)` (increases perceived complexity) or `(-)` (decreases perceived complexity).

Extract a scene graph (objects, attributes, relationships) that is **strictly grounded in the phrases**. Do not invent elements the phrases do not mention or imply. Use the `(+)` / `(-)` markers to inform the `sentiment` field of attributes and relationships.

{TOPICS_BLOCK}

{OUTPUT_SCHEMA_BLOCK}"""

SYSTEM_V1 = f"""You are a visual complexity annotation expert. You will receive, for a single data-visualization image:
  (a) the rendered image.

Extract a scene graph (objects, attributes, relationships) that is **strictly grounded in what is visible in the image**. Do not invent elements that are not visible.

{TOPICS_BLOCK}

{OUTPUT_SCHEMA_BLOCK}"""


# ── Helpers ───────────────────────────────────────────────────────────────
def load_rows() -> list[dict]:
    df = pd.read_csv(DATA_CSV)
    df9 = df[df['VisType'].isin(NINE_VISTYPES)].copy()
    assert len(df9) == 510, f'Expected 510, got {len(df9)}'
    keep = ['imageName', 'VisType', 'NormalizedVC', 'originalPhrases', 'originalSentiments']
    return df9[keep].to_dict('records')


def format_tagged_phrases(row: dict) -> str:
    phrases = (row.get('originalPhrases') or '').strip()
    sents   = (row.get('originalSentiments') or '').strip()
    if not phrases:
        return '(no phrases available)'
    p_list = [p.strip() for p in phrases.split(';') if p.strip()]
    s_list = [s.strip() for s in sents.split(';')] if sents else []
    pairs  = []
    for i, p in enumerate(p_list):
        s = s_list[i] if i < len(s_list) and s_list[i] else '(?)'
        pairs.append(f'- {p} {s}')
    return '\n'.join(pairs)


def parse_json(raw: str) -> dict:
    s = raw.strip()
    if s.startswith('```'):
        s = s.split('```')[1]
        if s.startswith('json'):
            s = s[4:]
    return json.loads(s)


def load_existing(path: Path) -> dict:
    # Warm-start: also pull in the 63/66-image eval files
    data = {}
    # Load from the original small eval file first (B=66, V1=63)
    stem = path.stem  # e.g. 'oar_B_510'
    cond = stem.split('_')[1]  # 'B' or 'V1'
    small_file = TC_DIR / f'oar_{cond}.json'
    if small_file.exists():
        data.update(json.loads(small_file.read_text(encoding='utf-8')))
        print(f'  Warm-start: loaded {len(data)} entries from {small_file.name}')
    # Then load any partial 510 run
    if path.exists():
        partial = json.loads(path.read_text(encoding='utf-8'))
        data.update(partial)
        print(f'  Warm-start: merged {len(partial)} entries from {path.name} → total {len(data)}')
    return data


def save(data: dict, path: Path):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


async def fetch_image_base64(session: aiohttp.ClientSession, image_name: str) -> str:
    url = IMAGE_BASE_URL + image_name
    async with session.get(url) as resp:
        if resp.status != 200:
            raise RuntimeError(f'HTTP {resp.status} for {url}')
        return base64.standard_b64encode(await resp.read()).decode('utf-8')


# ── Per-condition extraction ──────────────────────────────────────────────
async def extract_one(client, session, row: dict, cond: str, system_text: str) -> dict:
    if cond == 'B':
        phrases = format_tagged_phrases(row)
        user_content = (
            f"Image: {row['imageName']}\n\n"
            f"Original complexity phrases extracted from participant comments "
            f"(each tagged with sentiment: (+) increases complexity, (-) decreases complexity):\n"
            f"{phrases}\n\n"
            f"Extract the scene graph grounded in these phrases."
        )
        messages = [{'role': 'user', 'content': user_content}]
    elif cond == 'V1':
        b64 = await fetch_image_base64(session, row['imageName'])
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': b64}},
                {'type': 'text',  'text': f"Image: {row['imageName']}\n\nExtract the scene graph grounded in what you see."},
            ],
        }]
    else:
        raise ValueError(f'Unknown condition: {cond}')

    system_block = [{'type': 'text', 'text': system_text, 'cache_control': {'type': 'ephemeral'}}]
    resp = await client.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
        system=system_block, messages=messages,
    )
    return parse_json(resp.content[0].text)


async def run_condition(cond: str, args):
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print('ERROR: ANTHROPIC_API_KEY not set'); sys.exit(1)

    client      = anthropic.AsyncAnthropic(api_key=api_key)
    system_text = SYSTEM_B if cond == 'B' else SYSTEM_V1
    out_file    = TC_DIR / f'oar_{cond}_510.json'

    rows     = load_rows()
    existing = load_existing(out_file)

    if args.limit:
        rows = rows[:args.limit]

    to_do = [r for r in rows if r['imageName'] not in existing]
    print(f'\n[{cond}] total={len(rows)}  already_done={len(existing)}  to_do={len(to_do)}')

    if not to_do:
        print(f'[{cond}] Nothing to do — saving merged file.')
        save(existing, out_file)
        return

    sem   = asyncio.Semaphore(args.concurrency or CONCURRENCY)
    lock  = asyncio.Lock()
    cnt   = {'ok': 0, 'fail': 0, 'n': len(to_do)}

    async with aiohttp.ClientSession() as session:
        async def worker(row):
            async with sem:
                name = row['imageName']
                try:
                    ext = await extract_one(client, session, row, cond, system_text)
                    async with lock:
                        existing[name] = ext
                        cnt['ok'] += 1
                        if cnt['ok'] % 10 == 0:
                            save(existing, out_file)   # checkpoint every 10
                    done = cnt['ok'] + cnt['fail']
                    print(f"  [{done}/{cnt['n']}] {cond} {name} OK "
                          f"({len(ext.get('objects',[]))}obj "
                          f"{len(ext.get('attributes',[]))}attr "
                          f"{len(ext.get('relationships',[]))}rel)")
                except Exception as e:
                    async with lock:
                        cnt['fail'] += 1
                    print(f"  [{cnt['ok']+cnt['fail']}/{cnt['n']}] {cond} {name} ERROR: {e}")

        await asyncio.gather(*[worker(r) for r in to_do])

    save(existing, out_file)
    print(f'\n[{cond}] Finished. ok={cnt["ok"]} fail={cnt["fail"]} total_saved={len(existing)} → {out_file}')


def main():
    ap = argparse.ArgumentParser(description='Extract OAR for all 510 images (B and V1 conditions).')
    ap.add_argument('--condition', choices=['B', 'V1', 'all'], required=True)
    ap.add_argument('--limit',       type=int, default=None,  help='Process only first N images (smoke test)')
    ap.add_argument('--concurrency', type=int, default=None,  help='Async concurrency limit (default 5)')
    args = ap.parse_args()

    conds = ['B', 'V1'] if args.condition == 'all' else [args.condition]
    for c in conds:
        asyncio.run(run_condition(c, args))


if __name__ == '__main__':
    main()
