"""
Quick single-image comparison of three V3 (phrases + image) system-prompt framings.
=========================================================================
Runs all three framings against whoO06_2.png (or any image via --image),
prints a side-by-side comparison with the input phrases, and saves raw JSON.

Usage:
  python scripts/_test_v3_variants.py
  python scripts/_test_v3_variants.py --image InfoVisJ.619.17.png
"""
from __future__ import annotations
import os, sys, json, asyncio, argparse, base64, textwrap
from pathlib import Path
import aiohttp
import pandas as pd
from dotenv import load_dotenv
import anthropic

ROOT = Path(__file__).parent.parent.parent
load_dotenv(dotenv_path=ROOT.parent / '.env')

MODEL           = 'claude-sonnet-4-6'
MAX_TOKENS      = 2048
TEMPERATURE     = 0.0
IMAGE_BASE_URL  = 'https://raw.githubusercontent.com/c109363/ExperimentImage/main/AllDataResize/'
DATA_CSV        = ROOT / 'phrase_reduction_v2' / 'image_compiled_phrases.csv'
OUT_DIR         = ROOT / 'vc_genome_output_full' / 'three_conditions' / 'v3_probe'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_IMAGE   = 'whoO06_2.png'

# ── Shared blocks (copied from _extract_three_conditions.py) ─────────────
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

# ── Three V3 system prompts ───────────────────────────────────────────────
SYSTEMS = {
    'V3-hard': f"""You are a visual complexity annotation expert. You will receive, for a single data-visualization image:
  (a) a list of original complexity phrases extracted from real participant comments about the image, each tagged with a sentiment marker `(+)` (increases perceived complexity) or `(-)` (decreases perceived complexity).
  (b) the rendered image.

Extract a scene graph **strictly grounded in the phrases**. The phrases are your primary evidence; do not add elements they do not mention or imply. Use the image only to resolve spatial ambiguity (e.g., which region an object belongs to) or to clarify how a mentioned object appears visually.

{TOPICS_BLOCK}

{OUTPUT_SCHEMA_BLOCK}""",

    'V3-soft': f"""You are a visual complexity annotation expert. You will receive, for a single data-visualization image:
  (a) a list of original complexity phrases extracted from real participant comments about the image, each tagged with a sentiment marker `(+)` (increases perceived complexity) or `(-)` (decreases perceived complexity).
  (b) the rendered image.

Extract a scene graph that integrates both sources. The phrases reveal which complexity aspects human observers noticed and carry ground-truth sentiment; the image provides spatial and visual grounding. Prioritize elements mentioned in the phrases, but you may also include visually prominent elements that clearly contribute to complexity even if not explicitly named in the phrases.

{TOPICS_BLOCK}

{OUTPUT_SCHEMA_BLOCK}""",

    'V3-hint': f"""You are a visual complexity annotation expert. You will receive, for a single data-visualization image:
  (a) a list of original complexity phrases extracted from real participant comments about the image, each tagged with a sentiment marker `(+)` (increases perceived complexity) or `(-)` (decreases perceived complexity).
  (b) the rendered image.

Extract a scene graph **grounded in what is visible in the image**. The phrases are provided as a reference showing which complexity aspects human observers noticed — use them to bias your attention and to inform sentiment, but base your extraction primarily on visual evidence and do not restrict yourself to what the phrases mention.

{TOPICS_BLOCK}

{OUTPUT_SCHEMA_BLOCK}""",
}


# ── Helpers ───────────────────────────────────────────────────────────────
def format_tagged_phrases(row) -> str:
    phrases = (row.get('originalPhrases') or '').strip()
    sents   = (row.get('originalSentiments') or '').strip()
    if not phrases:
        return '(none)'
    p_list = [p.strip() for p in phrases.split(';') if p.strip()]
    s_list = [s.strip() for s in sents.split(';')] if sents else []
    pairs = []
    for i, p in enumerate(p_list):
        s = s_list[i] if i < len(s_list) and s_list[i] else '(?)'
        pairs.append(f'  - {p} {s}')
    return '\n'.join(pairs)


def format_phrases_only(row) -> str:
    """Phrases without sentiment tags."""
    phrases = (row.get('originalPhrases') or '').strip()
    if not phrases:
        return '(none)'
    return '\n'.join(f'  - {p.strip()}' for p in phrases.split(';') if p.strip())


def parse_json(raw: str) -> dict:
    s = raw.strip()
    if s.startswith('```'):
        s = s.split('```')[1]
        if s.startswith('json'):
            s = s[4:]
    return json.loads(s)


async def fetch_image_base64(session: aiohttp.ClientSession, image_name: str) -> str:
    url = IMAGE_BASE_URL + image_name
    async with session.get(url) as resp:
        if resp.status != 200:
            raise RuntimeError(f'HTTP {resp.status} fetching {url}')
        return base64.standard_b64encode(await resp.read()).decode('utf-8')


async def run_one_variant(client, image_name: str, b64: str, row: dict,
                          variant: str, system_text: str, no_sentiment: bool = False) -> dict:
    phrases_text = format_phrases_only(row) if no_sentiment else format_tagged_phrases(row)
    sentiment_label = 'phrases only (no sentiment tags)' if no_sentiment else 'phrases with sentiment tags'
    user_content = [
        {'type': 'image',  'source': {'type': 'base64', 'media_type': 'image/png', 'data': b64}},
        {'type': 'text',   'text': (
            f"Image: {image_name}\n\n"
            f"Original complexity phrases from participant comments ({sentiment_label}):\n"
            f"{phrases_text}\n\n"
            f"Extract the scene graph."
        )},
    ]
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=[{'type': 'text', 'text': system_text}],
        messages=[{'role': 'user', 'content': user_content}],
    )
    return parse_json(resp.content[0].text)


# ── Printing ──────────────────────────────────────────────────────────────
COL = 52          # width per variant column
SEP = '  |  '

def _wrap(text: str, width: int = COL) -> list[str]:
    return textwrap.wrap(str(text), width) or ['']

def _cell(text: str, width: int = COL) -> str:
    return str(text).ljust(width)[:width]

def print_section_header(title: str):
    bar = '-' * (COL * 3 + len(SEP) * 2)
    print(f'\n{"----"} {title} {"-"*(len(bar)-6-len(title))}')

def print_objects(results: dict[str, dict]):
    print_section_header('OBJECTS')
    header = SEP.join(_cell(f'{v}  (objects={len(d.get("objects",[]))})', COL)
                      for v, d in results.items())
    print(header)
    print(SEP.join(['-' * COL] * 3))

    # collect per-variant rows
    obj_rows = {v: d.get('objects', []) for v, d in results.items()}
    max_rows = max(len(r) for r in obj_rows.values())
    for i in range(max_rows):
        cells = []
        for v in results:
            objs = obj_rows[v]
            if i < len(objs):
                o = objs[i]
                cells.append(_cell(f"[{o['id']}] {o['name']}  ({o.get('region','')})", COL))
            else:
                cells.append(' ' * COL)
        print(SEP.join(cells))

def print_attributes(results: dict[str, dict]):
    print_section_header('ATTRIBUTES')
    header = SEP.join(_cell(f'{v}  (attrs={len(d.get("attributes",[]))})', COL)
                      for v, d in results.items())
    print(header)
    print(SEP.join(['-' * COL] * 3))

    attr_rows = {v: d.get('attributes', []) for v, d in results.items()}
    max_rows = max(len(r) for r in attr_rows.values())
    for i in range(max_rows):
        cells = []
        for v in results:
            attrs = attr_rows[v]
            if i < len(attrs):
                a = attrs[i]
                topic_short = a.get('topic','').split('/')[0].strip()[:20]
                cells.append(_cell(
                    f"obj{a['object_id']} {a['sentiment']}  {a['attr']}  [{topic_short}]", COL))
            else:
                cells.append(' ' * COL)
        print(SEP.join(cells))

def print_relationships(results: dict[str, dict]):
    print_section_header('RELATIONSHIPS')
    header = SEP.join(_cell(f'{v}  (rels={len(d.get("relationships",[]))})', COL)
                      for v, d in results.items())
    print(header)
    print(SEP.join(['-' * COL] * 3))

    rel_rows = {v: d.get('relationships', []) for v, d in results.items()}
    max_rows = max(len(r) for r in rel_rows.values())
    for i in range(max_rows):
        cells = []
        for v in results:
            rels = rel_rows[v]
            if i < len(rels):
                r = rels[i]
                topic_short = r.get('topic','').split('/')[0].strip()[:14]
                cells.append(_cell(
                    f"obj{r['subj']} -{r['pred']}-> obj{r['obj']}  {r['sentiment']}  [{topic_short}]",
                    COL))
            else:
                cells.append(' ' * COL)
        print(SEP.join(cells))


# ── Main ──────────────────────────────────────────────────────────────────
async def main(image_name: str, args):
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print('ERROR: ANTHROPIC_API_KEY not set'); sys.exit(1)

    # Load phrases
    df = pd.read_csv(DATA_CSV)
    row_match = df[df['imageName'] == image_name]
    if row_match.empty:
        print(f'ERROR: {image_name} not found in {DATA_CSV}'); sys.exit(1)
    row = row_match.iloc[0].to_dict()

    print(f'\n{"="*80}')
    print(f'  V3 Variant Probe  —  {image_name}')
    print(f'{"="*80}')
    print(f'\nInput phrases (sentiment {"HIDDEN" if args.no_sentiment else "PROVIDED"}):')
    print(format_phrases_only(row) if args.no_sentiment else format_tagged_phrases(row))
    print(f'\nVC score (NormalizedVC): {row.get("NormalizedVC","n/a")}')
    print(f'VisType: {row.get("VisType","n/a")}')

    client = anthropic.AsyncAnthropic(api_key=api_key)

    async with aiohttp.ClientSession() as session:
        print(f'\nFetching image...')
        b64 = await fetch_image_base64(session, image_name)
        print(f'Running {len(SYSTEMS)} variants in parallel...')
        tasks = {
            v: run_one_variant(client, image_name, b64, row, v, sys_text, no_sentiment=args.no_sentiment)
            for v, sys_text in SYSTEMS.items()
        }
        results_list = await asyncio.gather(*tasks.values(), return_exceptions=True)

    results: dict[str, dict] = {}
    for variant, result in zip(tasks.keys(), results_list):
        if isinstance(result, Exception):
            print(f'  {variant}: ERROR — {result}')
            results[variant] = {'objects': [], 'attributes': [], 'relationships': []}
        else:
            results[variant] = result
            print(f'  {variant}: {len(result.get("objects",[]))} obj, '
                  f'{len(result.get("attributes",[]))} attr, '
                  f'{len(result.get("relationships",[]))} rel')

    # Save raw JSON
    suffix = '_nosent' if args.no_sentiment else ''
    out_file = OUT_DIR / f'{Path(image_name).stem}_v3_variants{suffix}.json'
    out_file.write_text(json.dumps(
        {'image': image_name, 'phrases': row.get('originalPhrases',''),
         'sentiments': row.get('originalSentiments',''),
         'sentiment_provided': not args.no_sentiment,
         'results': results},
        indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nRaw JSON saved -> {out_file}')

    # Print comparison
    print_objects(results)
    print_attributes(results)
    print_relationships(results)
    print(f'\n{"="*80}\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default=DEFAULT_IMAGE,
                    help=f'Image filename (default: {DEFAULT_IMAGE})')
    ap.add_argument('--no-sentiment', action='store_true',
                    help='Strip sentiment tags from phrases before sending to the model')
    args = ap.parse_args()
    asyncio.run(main(args.image, args))
