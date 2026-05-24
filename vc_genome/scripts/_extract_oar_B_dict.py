"""
B_dict condition: OAR extraction using human-curated vocabulary.
=================================================================
Differences from B condition:
  - Object names guided by image-specific `objectWords` (soft constraint)
  - Attributes/relationships tagged with image-specific subtopics (hard)
  - No sentiment in input or output

Test set: 9 images sampled from phrase_reduction_v2/outputs/image_compiled_phrases.csv,
one image randomly drawn from each 0.1 NormalizedVC bin.

Usage:
    python scripts/_extract_oar_B_dict.py [--seed 42] [--dry-run]

Outputs:
    vc_genome_output_full/three_conditions/oar_B_dict_9.json
    vc_genome_output_full/three_conditions/B_dict_sample_9.csv
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

import pandas as pd
import anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent.parent
load_dotenv(dotenv_path=ROOT.parent / '.env')

PHRASE_CSV     = ROOT / 'phrase_reduction_v2' / 'outputs' / 'image_compiled_phrases.csv'
SHORTLIST_CSV  = ROOT / 'phrase_reduction_v2' / 'outputs' / 'phrase_shortlist.csv'
TC_DIR         = ROOT / 'vc_genome_output_full' / 'three_conditions'
TC_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = TC_DIR / 'oar_B_dict_9.json'
OUT_CSV  = TC_DIR / 'B_dict_sample_9.csv'

MODEL       = 'claude-sonnet-4-6'
MAX_TOKENS  = 2048
TEMPERATURE = 0.0

TOPICS = [
    'Data Density / Image Clutter',
    'Visual Encoding Clarity',
    'Semantics / Text Legibility',
    'Schema',
    'Color, Symbol, and Texture Details',
    'Aesthetics Uncertainty',
    'Immediacy / Cognitive Load',
]

REGIONS = 'data_area | x_axis | y_axis | legend | title | annotation | colorbar | background | overall'


# ── Build subtopic descriptions block from phrase_shortlist.csv ───────────
def build_subtopics_block() -> str:
    df = pd.read_csv(SHORTLIST_CSV)
    lines = ['## Subtopic Reference\n']
    cur_topic = None
    for _, row in df.iterrows():
        topic = str(row['Topic']).strip()
        sub   = str(row['SubTopic']).strip()
        desc  = str(row['Description']).strip().replace('\n', ' ')
        if topic != cur_topic:
            lines.append(f'\n### {topic}')
            cur_topic = topic
        lines.append(f'- **{sub}**: {desc}')
    return '\n'.join(lines)


SUBTOPICS_BLOCK = build_subtopics_block()

OUTPUT_SCHEMA_BLOCK = f"""## Output Schema

Return ONLY valid JSON (no markdown fences, no prose outside JSON):
{{
  "objects": [
    {{"id": 1, "name": "object_name", "region": "{REGIONS}"}}
  ],
  "attributes": [
    {{"object_id": 1, "attr": "short_snake_case_phrase", "topic": "<one of the 7 topics>", "subtopic": "<one from this image's active subtopics>"}}
  ],
  "relationships": [
    {{"subj": 1, "pred": "snake_case_predicate", "obj": 2, "topic": "<one of the 7 topics>", "subtopic": "<one from this image's active subtopics>"}}
  ]
}}

Guidelines:
- Extract elements proportional to the richness of the input. Typical ranges: 2–6 objects, 3–8 attributes, 1–5 relationships.
- Each list must contain at least one entry.
- Object names: single lowercase word or snake_case, preferably from the vocabulary provided.
  If no vocabulary term fits, use the most specific snake_case name you can — do not use "unknown".
- Attribute text: snake_case, max 4 words.
- topic must be exactly one of the 7 topic titles listed above.
- subtopic must be exactly one from the active subtopics list provided in the user message.
- No sentiment field."""

TOPICS_BLOCK = f"""## Topics (7-topic taxonomy)

Tag each attribute and relationship with exactly one topic:

1. **Data Density / Image Clutter** — Perceived amount, richness, or depth of data; information volume, element quantity, and visual clutter/overlap.
2. **Visual Encoding Clarity** — Variety, type, and complexity of graphical forms; spatial layout, scale, and encoding interpretability.
3. **Semantics / Text Legibility** — Quantity and density of text elements (titles, axis labels, legends, captions, annotations).
4. **Schema** — Whether specialized domain knowledge is needed; dimensionality, structural complexity, abstraction level.
5. **Color, Symbol, and Texture Details** — Range, variety, and arrangement of colors; use of symbols, textures, and non-color graphical markers.
6. **Aesthetics Uncertainty** — How visually cluttered, dense, or disordered the layout appears.
7. **Immediacy / Cognitive Load** — Overall ease or difficulty of interpreting the visualization."""

SYSTEM_B_DICT = f"""You are a visual complexity annotation expert. You will receive, for a single \
data-visualization image, a list of complexity phrases extracted from real participant comments.

Extract a scene graph (objects, attributes, relationships) that is **strictly grounded in the \
phrases**. Do not invent elements the phrases do not mention or imply.

{TOPICS_BLOCK}

{SUBTOPICS_BLOCK}

{OUTPUT_SCHEMA_BLOCK}"""


# ── Sampling ──────────────────────────────────────────────────────────────
def sample_9(seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(PHRASE_CSV)
    bins = [i / 10 for i in range(0, 11)]
    df['_bin'] = pd.cut(df['NormalizedVC'], bins=bins, include_lowest=True)
    sampled = (
        df.groupby('_bin', observed=True)
          .apply(lambda g: g.sample(n=1, random_state=seed), include_groups=False)
          .reset_index(level=0)
          .rename(columns={'_bin': 'vc_bin'})
          .reset_index(drop=True)
    )
    sampled = sampled.sort_values('NormalizedVC').reset_index(drop=True)
    print(f'Sampled {len(sampled)} images:')
    for _, r in sampled.iterrows():
        print(f'  [{r["vc_bin"]}]  {r["imageName"]:30s}  VC={r["NormalizedVC"]:.2f}  {r["VisType"]}')
    return sampled


# ── Prompt helpers ─────────────────────────────────────────────────────────
def strip_sentiment(phrases_str: str) -> list[str]:
    """Remove (+) / (-) markers and return clean phrase list."""
    raw = [p.strip() for p in str(phrases_str).split(';') if p.strip()]
    return [re.sub(r'\s*\([+-]\)\s*$', '', p).strip() for p in raw]


def format_object_words(obj_words_str: str) -> str:
    words = [w.strip() for w in str(obj_words_str).split(';') if w.strip()]
    return ', '.join(words) if words else '(none specified)'


def format_subtopics(subtopics_str: str) -> str:
    subs = [s.strip() for s in str(subtopics_str).split(';') if s.strip()]
    return '\n'.join(f'  - {s}' for s in subs)


def build_user_message(row: pd.Series) -> str:
    phrases = strip_sentiment(row.get('originalPhrases', ''))
    phrase_block = '\n'.join(f'- {p}' for p in phrases) if phrases else '(no phrases available)'

    obj_vocab = format_object_words(row.get('objectWords', ''))
    subtopics = format_subtopics(row.get('SubTopics', ''))

    return (
        f"Image: {row['imageName']}\n\n"
        f"Complexity phrases from participant comments:\n{phrase_block}\n\n"
        f"Preferred object vocabulary for this image (use these when they fit):\n"
        f"  {obj_vocab}\n\n"
        f"Active subtopics for this image — tag every attribute and relationship "
        f"with exactly one of:\n{subtopics}\n\n"
        f"Extract the scene graph grounded in these phrases."
    )


# ── Extraction ────────────────────────────────────────────────────────────
def parse_json(raw: str) -> dict:
    s = raw.strip()
    if s.startswith('```'):
        s = s.split('```')[1]
        if s.startswith('json'):
            s = s[4:]
    return json.loads(s)


def extract_one(client: anthropic.Anthropic, row: pd.Series, dry_run: bool) -> dict:
    user_msg = build_user_message(row)
    if dry_run:
        print(f'\n--- DRY RUN: {row["imageName"]} ---')
        print('SYSTEM (first 300 chars):', SYSTEM_B_DICT[:300], '...')
        print('USER:\n', user_msg)
        return {'objects': [], 'attributes': [], 'relationships': []}

    system_block = [{'type': 'text', 'text': SYSTEM_B_DICT,
                     'cache_control': {'type': 'ephemeral'}}]
    resp = client.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
        system=system_block,
        messages=[{'role': 'user', 'content': user_msg}],
    )
    return parse_json(resp.content[0].text)


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--dry-run', action='store_true',
                        help='Print prompts without calling the API')
    args = parser.parse_args()

    sample = sample_9(seed=args.seed)
    sample.to_csv(OUT_CSV, index=False)
    print(f'\nSample saved → {OUT_CSV.relative_to(ROOT.parent)}')

    client  = anthropic.Anthropic()
    results = {}

    for i, row in sample.iterrows():
        name = row['imageName']
        print(f'\n[{i+1}/9] Extracting {name} …')
        try:
            oar = extract_one(client, row, dry_run=args.dry_run)
            results[name] = oar
            if not args.dry_run:
                print(f'  obj={len(oar["objects"])}  '
                      f'attr={len(oar["attributes"])}  '
                      f'rel={len(oar["relationships"])}')
        except Exception as exc:
            print(f'  ERROR: {exc}', file=sys.stderr)
            results[name] = {'error': str(exc)}

    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                        encoding='utf-8')
    print(f'\nDone. Saved {len(results)} entries → {OUT_JSON.relative_to(ROOT.parent)}')


if __name__ == '__main__':
    main()
