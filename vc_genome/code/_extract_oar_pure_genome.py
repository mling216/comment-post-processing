"""
_extract_oar_pure_genome.py
===========================
OAR extraction — pure-genome condition.

Uses a global shared vocabulary (pure_genome_dict.json) built from the full corpus.
No per-image topic or subtopic constraints — vocabulary is the only grounding.

Input phrases:  CuratePhrasesMore/LessComplex column
                (ResultsStepByStep_4.0.imageDataCompiled.csv)
Vocabulary:     vc_genome/export/pure_genome_dict.json

Modes
-----
  --mode sample   Same 9 images as the curate_dict pilot sample  [default]
                  (loaded from oar_curate_dict_9.json key order)
  --mode full     All ~497 images in the compiled CSV (9 VisTypes)

Options
-------
  --limit N       Process only first N images           (smoke test)
  --dry-run       Print prompt for first image; no API calls
  --out FILE      Override output path

Output
------
  sample  →  vc_genome/export/oar_pure_genome_9.json
  full    →  vc_genome/export/oar_pure_genome_full.json

Usage
-----
  python code/_extract_oar_pure_genome.py
  python code/_extract_oar_pure_genome.py --mode full
  python code/_extract_oar_pure_genome.py --dry-run
  python code/_extract_oar_pure_genome.py --limit 3
"""
from __future__ import annotations
import os, sys, json, asyncio, argparse, math
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import anthropic

SCRIPT_DIR = Path(__file__).parent
ROOT       = SCRIPT_DIR.parent.parent          # comment_post_processing/
load_dotenv(dotenv_path=ROOT.parent / '.env')

MODEL       = 'claude-sonnet-4-6'
MAX_TOKENS  = 2048
TEMPERATURE = 0.0
CONCURRENCY = 5

INPUT_CSV   = ROOT / 'comment_process' / 'ResultsStepByStep_4.0.imageDataCompiled.csv'
EXPORT_DIR  = ROOT / 'vc_genome' / 'export'
DICT_JSON   = EXPORT_DIR / 'pure_genome_dict.json'

# curate_dict sample — used to pick the same 9 images in sample mode
CURATE_JSON = ROOT / 'vc_genome_output_full' / 'three_conditions' / 'oar_curate_dict_9.json'

PHRASE_COL    = 'CuratePhrasesMore/LessComplex'
NINE_VISTYPES = ['Area', 'Bar', 'Cont.-ColorPatn', 'Glyph', 'Grid',
                 'Line', 'Node-link', 'Point', 'Text']

REGION_VALUES = ('data_area', 'x_axis', 'y_axis', 'legend', 'title',
                 'annotation', 'colorbar', 'background', 'overall')


# ── Vocabulary prompt block ───────────────────────────────────────────────

def _build_vocab_section(d: dict) -> str:
    """Return a compact, human-readable vocabulary reference string."""
    lines = ['## Shared Vocabulary (use these terms preferentially)\n']

    for top_cat in ('visual_objects', 'visual_attributes', 'visual_predicates'):
        section = d.get(top_cat, {})
        header = {'visual_objects':    'OBJECTS — preferred names for visual elements',
                  'visual_attributes': 'ATTRIBUTES — descriptive phrases for object properties',
                  'visual_predicates': 'PREDICATES — verbs for relationships between objects',
                  }[top_cat]
        lines.append(f'### {header}')
        for subcat, terms in section.items():
            lines.append(f'  [{subcat}]  ' + ', '.join(terms))
        lines.append('')

    # Unclustered: just mention the pool with a sample
    unc = d.get('unclustered', {})
    unc_total = sum(len(v) for v in unc.values())
    sample_nouns = unc.get('noun', [])[:12]
    sample_vbs   = unc.get('verb', [])[:8]
    sample_adjs  = unc.get('adj', [])[:8]
    lines.append(
        f'### FALLBACK POOL ({unc_total} additional terms, count<3 in corpus)\n'
        f'  nouns (sample): {", ".join(sample_nouns)}\n'
        f'  verbs (sample): {", ".join(sample_vbs)}\n'
        f'  adjs  (sample): {", ".join(sample_adjs)}\n'
        f'  (Use any of these if the clustered vocabulary lacks a fitting term.)'
    )
    return '\n'.join(lines)


def build_system_prompt(genome_dict: dict) -> str:
    vocab_section = _build_vocab_section(genome_dict)
    return (
        'You are a visual complexity annotation expert. You will receive, for a '
        'single data-visualization image, complexity phrases extracted from real '
        'participant comments.\n\n'
        'Extract a scene graph (objects, attributes, relationships) that is '
        '**strictly grounded in the phrases**. Do not invent elements the phrases '
        'do not mention or imply.\n\n'
        + vocab_section +
        '\n\n'
        '## Output Schema\n'
        'Return ONLY valid JSON (no markdown fences, no prose outside JSON):\n'
        '{\n'
        '  "objects": [\n'
        '    {"id": 0, "name": "visualization", "region": "overall"},\n'
        '    {"id": 1, "name": "object_name",\n'
        '     "region": "data_area|x_axis|y_axis|legend|title|annotation|colorbar|background|overall"}\n'
        '  ],\n'
        '  "attributes": [\n'
        '    {"object_id": 0, "attr": "short_snake_case_phrase"}\n'
        '  ],\n'
        '  "relationships": [\n'
        '    {"subj": 1, "pred": "snake_case_predicate", "obj": 0}\n'
        '  ]\n'
        '}\n\n'
        'Guidelines:\n'
        '- Object id=0 ("visualization", region="overall") is a **mandatory anchor** '
        'that must always be the first entry in "objects". It represents the whole '
        'image and is the default attachment point for attributes that describe the '
        'visualization as a whole.\n'
        '- Only add objects with id>=1 for specific visual elements explicitly '
        'mentioned or clearly implied by the phrases. Do not invent objects to fill '
        'space — if the phrases name no specific element, use only id=0.\n'
        '- Object names (id>=1): prefer terms from the vocabulary above; use snake_case.\n'
        '- Attribute text: snake_case, max 4 words.\n'
        '- Relationship predicates: prefer verbs from the vocabulary above.\n'
        '- "attributes" and "relationships" may be empty lists if the phrases do not '
        'warrant them, but "objects" must always contain at least id=0.\n'
        '- region must be exactly one of the nine values listed above.'
    )


# ── User message ──────────────────────────────────────────────────────────

def _str(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ''
    return str(val).strip()


def format_user_message(row: dict) -> str:
    raw = _str(row.get(PHRASE_COL))
    phrases = [p.strip() for p in raw.split(';') if p.strip()]
    phrase_block = ('\n'.join(f'- {p}' for p in phrases)
                    if phrases else '(no phrases available)')
    return (
        f"Image: {row['imageName']}\n\n"
        f"Complexity phrases from participant comments:\n"
        f"{phrase_block}\n\n"
        f"Extract the scene graph grounded in these phrases."
    )


# ── JSON helpers ──────────────────────────────────────────────────────────

def parse_json(raw: str) -> dict:
    s = raw.strip()
    if s.startswith('```'):
        s = s.split('```')[1]
        if s.startswith('json'):
            s = s[4:]
    return json.loads(s.strip())


def load_existing(path: Path) -> dict:
    if path.exists():
        data = json.loads(path.read_text(encoding='utf-8'))
        print(f'  Warm-start: {len(data)} entries from {path.name}')
        return data
    return {}


def save(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


# ── Row selection ─────────────────────────────────────────────────────────

def get_sample_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return the same 9 images used in the curate_dict pilot (same order)."""
    if not CURATE_JSON.exists():
        print(f'ERROR: curate_dict JSON not found at {CURATE_JSON}', file=sys.stderr)
        sys.exit(1)
    curate_names = list(json.loads(CURATE_JSON.read_text(encoding='utf-8')).keys())
    sub = df[df['imageName'].isin(curate_names)].copy()
    # Preserve the order from curate_dict
    sub['_order'] = sub['imageName'].map({n: i for i, n in enumerate(curate_names)})
    sub = sub.sort_values('_order').drop(columns=['_order']).reset_index(drop=True)
    missing = [n for n in curate_names if n not in sub['imageName'].values]
    if missing:
        print(f'WARNING: {len(missing)} curate_dict images not found in compiled CSV: {missing}')
    print(f'Sample mode: using {len(sub)} images (same as curate_dict pilot):')
    for _, r in sub.iterrows():
        print(f'  [{r["NormalizedVC"]:.2f}]  {r["imageName"]}  ({r["VisType"]})')
    return sub


def get_full_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('NormalizedVC').reset_index(drop=True)
    print(f'Full mode: {len(df)} images')
    return df


# ── Async extraction ──────────────────────────────────────────────────────

async def extract_one(client, sem: asyncio.Semaphore,
                      row: dict, system: str, dry_run: bool) -> tuple[str, dict | None]:
    image_name = row['imageName']
    user_msg   = format_user_message(row)

    if dry_run:
        print(f'\n{"="*60}')
        print(f'DRY-RUN  image: {image_name}')
        print('--- SYSTEM (first 400 chars) ---')
        print(system[:400], '...')
        print('--- USER MESSAGE ---')
        print(user_msg)
        return image_name, None

    system_block = [{'type': 'text', 'text': system,
                     'cache_control': {'type': 'ephemeral'}}]
    messages = [{'role': 'user', 'content': user_msg}]

    async with sem:
        resp = await client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            system=system_block, messages=messages,
        )
    raw = resp.content[0].text
    result = parse_json(raw)
    n_obj  = len(result.get('objects', []))
    n_attr = len(result.get('attributes', []))
    n_rel  = len(result.get('relationships', []))
    print(f'  OK  {image_name}  obj={n_obj}  attr={n_attr}  rel={n_rel}')
    return image_name, result


async def run(rows: list[dict], existing: dict, out_path: Path,
              system: str, dry_run: bool) -> dict:
    todo = [r for r in rows if r['imageName'] not in existing]
    print(f'\n{len(rows)} images total, {len(existing)} already done, '
          f'{len(todo)} to extract.')
    if not todo:
        print('Nothing to do.')
        return existing

    client = anthropic.AsyncAnthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    sem    = asyncio.Semaphore(CONCURRENCY)

    tasks = [extract_one(client, sem, row, system, dry_run) for row in todo]
    results = await asyncio.gather(*tasks)

    data = dict(existing)
    for name, extraction in results:
        if extraction is not None:
            data[name] = extraction
            save(data, out_path)          # incremental save

    return data


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Extract OAR — pure-genome condition')
    ap.add_argument('--mode',    choices=['sample', 'full'], default='sample',
                    help='sample=same 9 images as curate_dict pilot; full=all images')
    ap.add_argument('--limit',   type=int, default=None,
                    help='Process only first N images (smoke test)')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print prompts without calling API')
    ap.add_argument('--out',     type=str, default=None,
                    help='Override output JSON path')
    args = ap.parse_args()

    # Load genome dict and build system prompt
    if not DICT_JSON.exists():
        print(f'ERROR: pure_genome_dict.json not found at {DICT_JSON}\n'
              f'Run build_genome_dict.py first.', file=sys.stderr)
        sys.exit(1)
    genome_dict = json.loads(DICT_JSON.read_text(encoding='utf-8'))
    system = build_system_prompt(genome_dict)
    print(f'Loaded pure_genome_dict.json  '
          f'(clustered: {sum(sum(len(v) for v in genome_dict[k].values()) for k in ("visual_objects","visual_attributes","visual_predicates"))} terms, '
          f'unclustered fallback: {sum(len(v) for v in genome_dict.get("unclustered",{}).values())} terms)')

    # Load data
    df = pd.read_csv(INPUT_CSV)
    df = df[df['VisType'].isin(NINE_VISTYPES)].copy()
    print(f'Loaded compiled CSV: {len(df)} images (9 VisTypes)')

    # Select rows
    if args.mode == 'sample':
        rows_df  = get_sample_rows(df)
        default_out = EXPORT_DIR / 'oar_pure_genome_9.json'
    else:
        rows_df  = get_full_rows(df)
        default_out = EXPORT_DIR / 'oar_pure_genome_full.json'

    if args.limit:
        rows_df = rows_df.head(args.limit)
        print(f'  (limited to first {args.limit} images)')

    out_path = Path(args.out) if args.out else default_out
    existing = {} if args.dry_run else load_existing(out_path)
    rows     = rows_df.to_dict('records')

    asyncio.run(run(rows, existing, out_path, system, dry_run=args.dry_run))

    if not args.dry_run:
        final = load_existing(out_path)
        print(f'\nDone — {len(final)} images saved → {out_path}')


if __name__ == '__main__':
    main()
