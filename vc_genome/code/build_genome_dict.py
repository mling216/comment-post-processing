"""
build_genome_dict.py
====================
Step 2 of the pure-genome pipeline.

Reads pure_genome_vocab_raw.csv (output of build_genome_vocab.py) and asks
Claude to group all terms into a hierarchical visualization vocabulary
dictionary (the "pure-genome dict").

The output JSON is used as the shared vocabulary constraint in the
pure-genome OAR extraction step.

Output
------
  vc_genome/export/pure_genome_dict.json
    {
      "visual_objects":     { "<subcategory>": ["term", ...], ... },
      "visual_attributes":  { "<subcategory>": ["phrase", ...], ... },
      "visual_predicates":  { "<subcategory>": ["verb", ...], ... }
    }

Usage
-----
  python code/build_genome_dict.py
  python code/build_genome_dict.py --min-count 2   (default: include count>=2)
  python code/build_genome_dict.py --dry-run
"""
from __future__ import annotations
import json, argparse, os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import anthropic

SCRIPT_DIR = Path(__file__).parent
ROOT       = SCRIPT_DIR.parent.parent

load_dotenv(dotenv_path=ROOT.parent / '.env')

EXPORT_DIR  = ROOT / 'vc_genome' / 'export'
IN_CSV      = EXPORT_DIR / 'pure_genome_vocab_raw.csv'
OUT_JSON    = EXPORT_DIR / 'pure_genome_dict.json'

MODEL       = 'claude-sonnet-4-6'
MAX_TOKENS  = 4096
TEMPERATURE = 0.0

SYSTEM = """\
You are an expert in data visualization, computational linguistics, and scene graph \
construction. You will receive a list of terms and phrases extracted from participant \
descriptions of data visualization images, grouped by their grammatical role \
(noun, verb, adjective, noun_phrase).

Your task: organize ALL provided terms into a hierarchical vocabulary dictionary \
for a visualization scene graph extraction system. This dictionary will be used as \
a shared vocabulary (like Visual Genome) for naming objects, writing attribute \
phrases, and selecting relationship predicates.

Output ONLY valid JSON (no markdown fences, no prose outside JSON) with this structure:
{
  "visual_objects": {
    "<subcategory>": ["term1", "term2", ...],
    ...
  },
  "visual_attributes": {
    "<subcategory>": ["phrase1", "phrase2", ...],
    ...
  },
  "visual_predicates": {
    "<subcategory>": ["verb1", "verb2", ...],
    ...
  }
}

Guidelines:
- visual_objects: named visual elements (marks, chart components, data elements, etc.)
  Suggested subcategories: mark, chart_component, annotation, data_element, chart_type, domain_object
- visual_attributes: descriptive properties (include multi-word phrases where meaningful)
  Suggested subcategories: complexity, readability, color, size_density, familiarity, structure, aesthetics
- visual_predicates: verbs/predicates for relationships between objects
  Suggested subcategories: spatial, semantic, structural, perceptual
- Group synonyms together under the most canonical/common term as the list entry.
- Include EVERY term from the input — do not discard any.
- Multi-word noun phrases belong in visual_objects if they name a thing, \
  or visual_attributes if they describe a quality.
- Keep subcategory names short, lowercase, snake_case.
- You may add or rename subcategories beyond the suggestions if the data warrants it.\
"""


def build_user_message(df: pd.DataFrame) -> str:
    sections = []
    for pos_type in ['noun', 'verb', 'adj', 'noun_phrase']:
        subset = df[df['pos_type'] == pos_type].sort_values('count', ascending=False)
        if subset.empty:
            continue
        label = {'noun': 'NOUNS', 'verb': 'VERBS', 'adj': 'ADJECTIVES',
                 'noun_phrase': 'NOUN PHRASES'}[pos_type]
        term_list = ', '.join(
            f'{r["term"]} ({r["count"]})'
            for _, r in subset.iterrows()
        )
        sections.append(f'## {label}\n{term_list}')

    return (
        f'Total terms: {len(df)}\n\n'
        + '\n\n'.join(sections)
        + '\n\nGroup all of the above terms into the vocabulary JSON.'
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster-min-count', type=int, default=3,
                    help='Min frequency for LLM clustering (default 3); '
                         'terms below threshold are preserved in "unclustered" section')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print prompt without calling API')
    args = ap.parse_args()

    df = pd.read_csv(IN_CSV)
    df_cluster   = df[df['count'] >= args.cluster_min_count].copy()
    df_uncluster = df[df['count'] <  args.cluster_min_count].copy()
    print(f'Loaded {len(df)} terms from {IN_CSV.name}')
    print(f'LLM-cluster (count>={args.cluster_min_count}): {len(df_cluster)} terms')
    print(f'Unclustered (count< {args.cluster_min_count}): {len(df_uncluster)} terms')
    print(df_cluster.groupby('pos_type')['term'].count().to_string())

    # Build unclustered section — flat sorted list per POS type
    unclustered: dict = {}
    for pos_type in ['noun', 'verb', 'adj', 'noun_phrase']:
        subset = df_uncluster[df_uncluster['pos_type'] == pos_type]
        if not subset.empty:
            unclustered[pos_type] = sorted(subset['term'].tolist())

    user_msg = build_user_message(df_cluster)

    if args.dry_run:
        print(f'\n{"="*60}\nSYSTEM (first 400 chars):\n{SYSTEM[:400]}...')
        print(f'\nUSER MESSAGE ({len(user_msg)} chars):\n{user_msg[:800]}...')
        print(f'\nUnclustered sample (noun[:10]): {unclustered.get("noun", [])[:10]}')
        return

    print(f'\nSending {len(df_cluster)} terms to {MODEL}...')
    client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=SYSTEM,
        messages=[{'role': 'user', 'content': user_msg}],
    )
    raw = resp.content[0].text.strip()

    # Strip markdown fences if present
    if raw.startswith('```'):
        raw = raw.split('```')[1]
        if raw.startswith('json'):
            raw = raw[4:]
        raw = raw.strip()

    genome_dict = json.loads(raw)

    # Inject unclustered section so nothing is lost
    genome_dict['unclustered'] = unclustered

    # Summary
    print('\nDictionary summary:')
    clustered_total = 0
    for top_cat, value in genome_dict.items():
        if top_cat == 'unclustered':
            total = sum(len(v) for v in value.values())
            print(f'  unclustered: {total} terms across {len(value)} pos types '
                  f'(fallback pool, count<{args.cluster_min_count})')
        else:
            total = sum(len(v) for v in value.values())
            clustered_total += total
            print(f'  {top_cat}: {len(value)} subcategories, {total} terms')
            for subcat, terms in value.items():
                print(f'    {subcat}: {len(terms)} terms  (e.g. {", ".join(terms[:4])})')
    unclustered_total = sum(len(v) for v in unclustered.values())
    print(f'\nTotal terms in dict: {clustered_total + unclustered_total} '
          f'(clustered: {clustered_total}, unclustered fallback: {unclustered_total})')

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(genome_dict, indent=2, ensure_ascii=False),
                        encoding='utf-8')
    print(f'\nSaved → {OUT_JSON}')


if __name__ == '__main__':
    main()
