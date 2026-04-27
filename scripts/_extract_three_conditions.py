"""
Three-condition O/A/R extraction for the paper's 63-image evaluation set.
=========================================================================
Shared:
  - Taxonomy: the 7 topics (NOT 21 subtopics).
  - Temperature: t=0.
  - Model: claude-sonnet-4-6.
  - Output JSON schema identical to Stage 1/2 so existing canonicalization
    and matching code applies.
  - Same schema field 'topic' (we name it 'topic' here instead of 'subtopic').
  - No VisType, no NormalizedVC in any prompt.

Conditions:
  B  — topics + human-curated phrases (no image)            [baseline, 66 imgs]
  V1 — topics + image (no phrases)                          [vision-only, 63 imgs]
  V2 — topics + image + 3 anchor exemplars                  [vision + anchors, 63 imgs]

For V2, the 3 anchor few-shot assistant turns are taken directly from the
B output (`oar_B.json`) for the three anchor images. This keeps the anchor
style consistent with the comparison reference (B).

Anchors are present in B (so we have their O/A/R for V2), but are NEVER
included in the 43-image evaluation comparisons.

Usage:
  python scripts/_extract_three_conditions.py --condition B
  python scripts/_extract_three_conditions.py --condition V1
  python scripts/_extract_three_conditions.py --condition V2
  python scripts/_extract_three_conditions.py --condition all
"""
from __future__ import annotations
import os, sys, json, asyncio, argparse, base64
from pathlib import Path
import aiohttp
import pandas as pd
from dotenv import load_dotenv
import anthropic

ROOT = Path(__file__).parent.parent
load_dotenv(dotenv_path=ROOT.parent / '.env')

MODEL       = 'claude-sonnet-4-6'
MAX_TOKENS  = 2048  # raised from 1024; complex images can exceed 1024-token output
TEMPERATURE = 0.0
CONCURRENCY = 5
IMAGE_BASE_URL = 'https://raw.githubusercontent.com/c109363/ExperimentImage/main/AllDataResize/'

DATA_CSV    = ROOT / 'phrase_reduction_v2' / 'image_compiled_phrases.csv'
EVAL_CSV    = ROOT / 'Claude_vc_prediction' / 'gt_all_66.csv'
OUT_DIR     = ROOT / 'vc_genome_output_full' / 'three_conditions'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANCHORS = [
    {'imageName': 'VisC.503.6.png',          'vc_score': 0.22},
    {'imageName': 'InfoVisJ.619.17.png',     'vc_score': 0.54},
    {'imageName': 'InfoVisJ.1149.6(1).png',  'vc_score': 0.95},
]
ANCHOR_NAMES = {a['imageName'] for a in ANCHORS}

# ── 7 Topics (verbatim from scripts/_vc_score_api_v0_topic_anchor.py) ────
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


# ── System prompts per condition ─────────────────────────────────────────
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

SYSTEM_V2 = SYSTEM_V1 + """

## Reference Anchor Examples

Below are three reference images with expert-produced scene graphs. Use them to calibrate your extraction style (granularity, object naming, predicate style, topic usage). Do NOT copy their content; produce an extraction for the target image shown last."""


# ── Helpers ──────────────────────────────────────────────────────────────
async def fetch_image_base64(session, image_name):
    url = IMAGE_BASE_URL + image_name
    async with session.get(url) as resp:
        if resp.status != 200:
            raise RuntimeError(f'HTTP {resp.status} for {url}')
        return base64.standard_b64encode(await resp.read()).decode('utf-8')


def load_eval_rows(include_anchors: bool):
    eval_df = pd.read_csv(EVAL_CSV)
    assert len(eval_df) == 66
    if not include_anchors:
        eval_df = eval_df[~eval_df['imageName'].isin(ANCHOR_NAMES)].copy()
        assert len(eval_df) == 63, f'expected 63, got {len(eval_df)}'
    data_df = pd.read_csv(DATA_CSV)
    keep = ['imageName', 'originalPhrases', 'originalSentiments']
    merged = eval_df.merge(data_df[keep], on='imageName', how='left')
    missing = merged['originalPhrases'].isna().sum()
    if missing:
        print(f'WARNING: {missing} rows lack originalPhrases')
    return merged.to_dict('records')


def format_tagged_phrases(row) -> str:
    """Pair `originalPhrases` with `originalSentiments` (parallel `; `-separated lists)."""
    phrases = (row.get('originalPhrases') or '').strip()
    sents = (row.get('originalSentiments') or '').strip()
    if not phrases:
        return ''
    p_list = [p.strip() for p in phrases.split(';') if p.strip()]
    s_list = [s.strip() for s in sents.split(';')] if sents else []
    pairs = []
    for i, p in enumerate(p_list):
        s = s_list[i] if i < len(s_list) and s_list[i] else '(?)'
        pairs.append(f'- {p} {s}')
    return '\n'.join(pairs)


def parse_json(raw):
    s = raw.strip()
    if s.startswith('```'):
        s = s.split('```')[1]
        if s.startswith('json'):
            s = s[4:]
    return json.loads(s)


def load_existing(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def save(data, path):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


# ── Message builders per condition ───────────────────────────────────────
def user_msg_B(row):
    phrases = format_tagged_phrases(row)
    return f"""Image: {row['imageName']}

Original complexity phrases extracted from participant comments for this image (each tagged with sentiment: (+) increases complexity, (-) decreases complexity):
{phrases}

Extract the scene graph grounded in these phrases."""


def user_content_V1(row, image_b64):
    return [
        {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': image_b64}},
        {'type': 'text',  'text': f"Image: {row['imageName']}\n\nExtract the scene graph grounded in what you see."},
    ]


def build_anchor_turns(anchor_oar: dict, anchor_b64_map: dict):
    """Few-shot user→assistant turns from B's anchor O/A/R."""
    turns = []
    for anchor in ANCHORS:
        name = anchor['imageName']
        if name not in anchor_oar or name not in anchor_b64_map:
            print(f'  WARNING: missing anchor O/A/R or image for {name}')
            continue
        turns.append({
            'role': 'user',
            'content': [
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': anchor_b64_map[name]}},
                {'type': 'text', 'text': f"Image: {name}\n\nExtract the scene graph grounded in what you see."},
            ],
        })
        turns.append({
            'role': 'assistant',
            'content': json.dumps(anchor_oar[name], ensure_ascii=False),
        })
    return turns


# ── Per-condition extractor ──────────────────────────────────────────────
async def extract_one(client, session, row, cond, system_text, anchor_turns):
    if cond == 'B':
        messages = [{'role': 'user', 'content': user_msg_B(row)}]
    elif cond == 'V1':
        b64 = await fetch_image_base64(session, row['imageName'])
        messages = [{'role': 'user', 'content': user_content_V1(row, b64)}]
    elif cond == 'V2':
        b64 = await fetch_image_base64(session, row['imageName'])
        messages = list(anchor_turns) + [{'role': 'user', 'content': user_content_V1(row, b64)}]
    else:
        raise ValueError(cond)

    system_block = [{'type': 'text', 'text': system_text, 'cache_control': {'type': 'ephemeral'}}]

    resp = await client.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
        system=system_block, messages=messages,
    )
    raw = resp.content[0].text
    return parse_json(raw)


async def run_condition(cond, args):
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print('ERROR: ANTHROPIC_API_KEY not set'); sys.exit(1)

    client = anthropic.AsyncAnthropic(api_key=api_key)
    # B is run on all 46 (its anchor outputs are needed for V2 few-shot).
    # V1 and V2 are run on the 43 eval images only.
    rows = load_eval_rows(include_anchors=(cond == 'B'))
    if args.limit:
        rows = rows[:args.limit]

    out_file = OUT_DIR / f'oar_{cond}.json'
    existing = {} if args.overwrite else load_existing(out_file)

    system_text = {'B': SYSTEM_B, 'V1': SYSTEM_V1, 'V2': SYSTEM_V2}[cond]

    anchor_turns = []
    if cond == 'V2':
        b_file = OUT_DIR / 'oar_B.json'
        if not b_file.exists():
            print(f'ERROR: {b_file} not found. Run --condition B first.')
            sys.exit(1)
        b_oar = json.loads(b_file.read_text(encoding='utf-8'))
        missing = [a['imageName'] for a in ANCHORS if a['imageName'] not in b_oar]
        if missing:
            print(f'ERROR: B output missing anchor entries: {missing}. Re-run --condition B.')
            sys.exit(1)
        async with aiohttp.ClientSession() as s:
            anchor_b64_map = {}
            for a in ANCHORS:
                anchor_b64_map[a['imageName']] = await fetch_image_base64(s, a['imageName'])
        anchor_turns = build_anchor_turns(b_oar, anchor_b64_map)
        print(f'  V2: {len(anchor_turns)//2} anchor turns sourced from oar_B.json')

    to_do = [r for r in rows if r['imageName'] not in existing]
    print(f'[{cond}] total={len(rows)}  done={len(existing)}  to_do={len(to_do)}')

    sem = asyncio.Semaphore(args.concurrency or CONCURRENCY)
    lock = asyncio.Lock()
    counter = {'ok': 0, 'fail': 0, 'total': len(to_do)}

    async with aiohttp.ClientSession() as session:
        async def worker(row):
            async with sem:
                name = row['imageName']
                try:
                    ext = await extract_one(client, session, row, cond, system_text, anchor_turns)
                    async with lock:
                        existing[name] = ext
                        counter['ok'] += 1
                        save(existing, out_file)
                    print(f"  [{counter['ok']+counter['fail']}/{counter['total']}] {cond} {name} OK "
                          f"({len(ext.get('objects',[]))} obj, {len(ext.get('attributes',[]))} attr, {len(ext.get('relationships',[]))} rel)")
                except Exception as e:
                    async with lock:
                        counter['fail'] += 1
                    print(f"  [{counter['ok']+counter['fail']}/{counter['total']}] {cond} {name} ERROR: {e}")

        await asyncio.gather(*[worker(r) for r in to_do])

    print(f'[{cond}] done. ok={counter["ok"]} fail={counter["fail"]}  →  {out_file}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', choices=['B', 'V1', 'V2', 'all'], required=True)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--concurrency', type=int, default=None)
    args = ap.parse_args()

    conds = ['B', 'V1', 'V2'] if args.condition == 'all' else [args.condition]
    for c in conds:
        asyncio.run(run_condition(c, args))


if __name__ == '__main__':
    main()
