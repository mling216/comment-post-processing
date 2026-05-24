"""
_extract_oar_curate_dict.py
===========================
OAR extraction — curate_dict condition.

Input grounding: human-curated phrases (no sentiment) + per-image objectWords
(preferred object vocabulary) + per-image SubTopics (required tagging ontology).

Modes
-----
  --mode sample   1 image per 0.1 NormalizedVC bin → 9 images  [default]
  --mode full     All images in image_compiled_phrases.csv

Options
-------
  --seed N        Random seed for bin sampling          (default 42)
  --limit N       Process only first N images           (smoke test)
  --dry-run       Print prompts for first image; no API calls
  --out FILE      Output JSON path (default: auto-named under three_conditions/)

Usage
-----
  python scripts/_extract_oar_curate_dict.py
  python scripts/_extract_oar_curate_dict.py --mode full
  python scripts/_extract_oar_curate_dict.py --dry-run
  python scripts/_extract_oar_curate_dict.py --limit 3
"""
from __future__ import annotations
import os, sys, json, asyncio, argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import anthropic

ROOT = Path(__file__).parent.parent.parent
load_dotenv(dotenv_path=ROOT.parent / '.env')

MODEL       = 'claude-sonnet-4-6'
MAX_TOKENS  = 2048
TEMPERATURE = 0.0
CONCURRENCY = 5

DATA_CSV = ROOT / 'phrase_reduction_v2' / 'outputs' / 'image_compiled_phrases.csv'
TC_DIR   = ROOT / 'vc_genome_output_full' / 'three_conditions'
TC_DIR.mkdir(parents=True, exist_ok=True)

NINE_VISTYPES = ['Area', 'Bar', 'Cont.-ColorPatn', 'Glyph', 'Grid',
                 'Line', 'Node-link', 'Point', 'Text']

# ── Prompt blocks ─────────────────────────────────────────────────────────

SYSTEM = """\
You are a visual complexity annotation expert. You will receive, for a single \
data-visualization image, complexity phrases extracted from real participant comments.

Extract a scene graph (objects, attributes, relationships) that is **strictly \
grounded in the phrases**. Do not invent elements the phrases do not mention or imply.

## Preferred Object Vocabulary
The user message includes preferred object names derived from participant \
descriptions of this specific image. Use terms from this list when they fit. \
If no term fits a visual element mentioned in the phrases, use the most specific \
snake_case name you can — do not use "unknown".

## Topics
Tag each attribute and relationship with exactly one of these 7 topics:
1. Data Density / Image Clutter
2. Visual Encoding Clarity
3. Semantics / Text Legibility
4. Schema
5. Color, Symbol, and Texture Details
6. Aesthetics Uncertainty
7. Immediacy / Cognitive Load

## Subtopics
The user message includes the specific subtopics assigned to this image from \
participant data. Tag each attribute and relationship with exactly one subtopic \
from that provided list.

## Output Schema
Return ONLY valid JSON (no markdown fences, no prose outside JSON):
{
  "objects": [
    {"id": 1, "name": "object_name",
     "region": "data_area|x_axis|y_axis|legend|title|annotation|colorbar|background|overall"}
  ],
  "attributes": [
    {"object_id": 1, "attr": "short_snake_case_phrase",
     "topic": "<one of the 7 topics>",
     "subtopic": "<one subtopic from the image list>"}
  ],
  "relationships": [
    {"subj": 1, "pred": "snake_case_predicate", "obj": 2,
     "topic": "<one of the 7 topics>",
     "subtopic": "<one subtopic from the image list>"}
  ]
}

Guidelines:
- Object names: single lowercase word or snake_case, preferably from the provided vocabulary.
- Attribute text: snake_case, max 4 words.
- Extract elements proportional to the richness of the input.
  Typical ranges: 2–6 objects, 3–8 attributes, 1–5 relationships.
- Each list must contain at least one entry.
- topic must be exactly one of the 7 topic titles listed above.
- subtopic must be exactly one from the list provided for this image.\
"""


def _str(val) -> str:
    """Safely coerce a potentially NaN field to string."""
    import math
    if val is None:
        return ''
    if isinstance(val, float) and math.isnan(val):
        return ''
    return str(val).strip()


def format_user_message(row: dict) -> str:
    # Phrases — no sentiment markers (strip (+) / (-) just in case)
    raw_phrases = _str(row.get('originalPhrases'))
    phrases = [p.strip() for p in raw_phrases.split(';') if p.strip()]
    phrase_block = '\n'.join(f'- {p}' for p in phrases) if phrases else '(no phrases available)'

    # Per-image preferred object words
    obj_words = _str(row.get('objectWords'))
    obj_list = [w.strip() for w in obj_words.split(';') if w.strip()] if obj_words else []
    obj_block = ', '.join(obj_list) if obj_list else '(none specified)'

    # Per-image subtopics
    subtopics_raw = _str(row.get('SubTopics'))
    subtopics = [s.strip() for s in subtopics_raw.split(';') if s.strip()]
    subtopic_block = '\n'.join(f'- {s}' for s in subtopics) if subtopics else '(none specified)'

    return (
        f"Image: {row['imageName']}\n\n"
        f"Preferred object vocabulary (from participant descriptions of this image):\n"
        f"{obj_block}\n\n"
        f"Subtopics for this image (use exactly these for tagging):\n"
        f"{subtopic_block}\n\n"
        f"Complexity phrases from participant comments:\n"
        f"{phrase_block}\n\n"
        f"Extract the scene graph grounded in these phrases."
    )


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
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


# ── Sampling ──────────────────────────────────────────────────────────────

def _row_is_valid(row: pd.Series) -> bool:
    """True if both objectWords and SubTopics are non-empty strings."""
    import math
    for col in ('objectWords', 'SubTopics'):
        v = row.get(col)
        if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == '':
            return False
    return True


def sample_rows(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """One random image per 0.1 NormalizedVC bin.
    Within each bin, prefer rows that have valid objectWords and SubTopics.
    Fall back to any row in the bin only if all rows in it are invalid.
    """
    bins = [i / 10 for i in range(0, 11)]
    df = df.copy()
    df['_bin'] = pd.cut(df['NormalizedVC'], bins=bins, include_lowest=True)

    picked = []
    for bin_label, group in df.groupby('_bin', observed=True):
        valid = group[group.apply(_row_is_valid, axis=1)]
        pool  = valid if len(valid) > 0 else group
        if len(pool) == 0:
            continue
        row = pool.sample(n=1, random_state=seed)
        if len(valid) == 0:
            print(f'  Warning: bin {bin_label} has no valid objectWords/SubTopics rows; '
                  f'using fallback.')
        picked.append(row)

    sampled = pd.concat(picked).reset_index(drop=True)
    if '_bin' in sampled.columns:
        sampled = sampled.drop(columns=['_bin'])
    sampled = sampled.sort_values('NormalizedVC').reset_index(drop=True)
    print(f'Sampled {len(sampled)} images (1 per 0.1 VC bin, seed={seed}):')
    for _, r in sampled.iterrows():
        valid_flag = '' if _row_is_valid(r) else '  [FALLBACK — no objectWords/SubTopics]'
        print(f'  [{r["NormalizedVC"]:.2f}]  {r["imageName"]}  ({r["VisType"]}){valid_flag}')
    return sampled


# ── Async extraction ──────────────────────────────────────────────────────

async def extract_one(client, sem: asyncio.Semaphore,
                      row: dict, dry_run: bool) -> tuple[str, dict | None]:
    image_name  = row['imageName']
    user_msg    = format_user_message(row)

    if dry_run:
        print(f'\n{"="*60}')
        print(f'DRY-RUN  image: {image_name}')
        print(f'--- SYSTEM (first 300 chars) ---')
        print(SYSTEM[:300], '...')
        print(f'--- USER MESSAGE ---')
        print(user_msg)
        return image_name, None

    system_block = [{'type': 'text', 'text': SYSTEM,
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
    print(f'  ✓  {image_name}  obj={n_obj}  attr={n_attr}  rel={n_rel}')
    return image_name, result


async def run(rows: list[dict], existing: dict,
              out_path: Path, dry_run: bool) -> dict:
    todo = [r for r in rows if r['imageName'] not in existing]
    print(f'{len(rows)} images total, {len(existing)} already done, '
          f'{len(todo)} to extract.')
    if not todo:
        print('Nothing to do.')
        return existing

    client = anthropic.AsyncAnthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    sem    = asyncio.Semaphore(CONCURRENCY)

    tasks = [extract_one(client, sem, row, dry_run) for row in todo]
    results = await asyncio.gather(*tasks)

    data = dict(existing)
    for name, extraction in results:
        if extraction is not None:
            data[name] = extraction
            save(data, out_path)          # incremental save after each batch

    return data


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Extract OAR — curate_dict condition')
    ap.add_argument('--mode',    choices=['sample', 'full'], default='sample')
    ap.add_argument('--seed',    type=int, default=42)
    ap.add_argument('--limit',   type=int, default=None)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--out',     type=str, default=None)
    args = ap.parse_args()

    df = pd.read_csv(DATA_CSV)
    df = df[df['VisType'].isin(NINE_VISTYPES)].copy()
    print(f'Using {len(df)} images from 9 main VisTypes (dropped {520 - len(df)} others)')

    if args.mode == 'sample':
        rows_df = sample_rows(df, seed=args.seed)
        default_out = TC_DIR / 'oar_curate_dict_9.json'
    else:
        rows_df = df.copy()
        default_out = TC_DIR / 'oar_curate_dict.json'

    if args.limit:
        rows_df = rows_df.head(args.limit)

    out_path = Path(args.out) if args.out else default_out
    existing = {} if args.dry_run else load_existing(out_path)
    rows     = rows_df.to_dict('records')

    asyncio.run(run(rows, existing, out_path, dry_run=args.dry_run))

    if not args.dry_run:
        print(f'\nSaved → {out_path}  ({len(existing) + len(rows)} entries)')


if __name__ == '__main__':
    main()
