"""
export_curate_dict_csv.py
=========================
Export curate_dict extraction results to CSV.

Output columns:
  imageName, imageURL, VisType, NormalizedVC,
  originalPhrases, Topics, SubTopics, objectWords, actionWords,
  attributes (extracted), relationships (extracted)

Usage
-----
  python scripts/export_curate_dict_csv.py
  python scripts/export_curate_dict_csv.py --input oar_curate_dict.json
  python scripts/export_curate_dict_csv.py --out my_output.csv
"""
from __future__ import annotations
import json, argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
ROOT       = SCRIPT_DIR.parent.parent          # comment_post_processing/

DATA_CSV   = ROOT / 'phrase_reduction_v2' / 'outputs' / 'image_compiled_phrases.csv'
TC_DIR      = ROOT / 'vc_genome_output_full' / 'three_conditions'
DEFAULT_IN  = TC_DIR / 'oar_curate_dict_9.json'
EXPORT_DIR  = ROOT / 'vc_genome' / 'export'
DEFAULT_OUT = EXPORT_DIR / 'curate_dict_9_sample_oar.csv'

DATA_COLS = ['imageName', 'imageURL', 'VisType', 'NormalizedVC',
             'originalPhrases', 'Topics', 'SubTopics', 'objectWords', 'actionWords']


def fmt_attributes(entry: dict) -> str:
    """Format extracted attributes as 'object: attr (subtopic)' joined by ' | '."""
    id_to_name = {o['id']: o['name'] for o in entry.get('objects', [])}
    parts = []
    for a in entry.get('attributes', []):
        obj = id_to_name.get(a.get('object_id'), '?')
        attr = a.get('attr', '').replace('_', ' ')
        sub  = a.get('subtopic', '')
        parts.append(f"{obj}: {attr} ({sub})" if sub else f"{obj}: {attr}")
    return ' | '.join(parts)


def fmt_relationships(entry: dict) -> str:
    """Format extracted relationships as 'subj pred obj (subtopic)' joined by ' | '."""
    id_to_name = {o['id']: o['name'] for o in entry.get('objects', [])}
    parts = []
    for r in entry.get('relationships', []):
        subj = id_to_name.get(r.get('subj'), '?')
        pred = r.get('pred', '').replace('_', ' ')
        obj  = id_to_name.get(r.get('obj'), '?')
        sub  = r.get('subtopic', '')
        parts.append(f"{subj} {pred} {obj} ({sub})" if sub else f"{subj} {pred} {obj}")
    return ' | '.join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(DEFAULT_IN),
                        help='Path to extraction JSON')
    parser.add_argument('--out', default=str(DEFAULT_OUT),
                        help='Output CSV path')
    args = parser.parse_args()

    json_path = Path(args.input)
    out_path  = Path(args.out)

    # Load extraction JSON
    with open(json_path, encoding='utf-8') as f:
        extraction: dict = json.load(f)
    print(f'Loaded {len(extraction)} images from {json_path.name}')

    # Load data CSV
    df_data = pd.read_csv(DATA_CSV, usecols=DATA_COLS)

    # Build rows
    rows = []
    for image_name, entry in extraction.items():
        meta = df_data[df_data['imageName'] == image_name]
        if meta.empty:
            print(f'  WARNING: {image_name} not found in data CSV — skipping')
            continue
        row = meta.iloc[0][DATA_COLS].to_dict()
        row['attributes']    = fmt_attributes(entry)
        row['relationships'] = fmt_relationships(entry)
        rows.append(row)

    # Assemble and save
    out_cols = DATA_COLS + ['attributes', 'relationships']
    df_out = pd.DataFrame(rows, columns=out_cols)
    df_out = df_out.rename(columns={
        'originalPhrases': 'originalPhrases\n(LLM input)',
        'SubTopics':        'SubTopics\n(LLM input)',
        'objectWords':      'objectWords\n(LLM input)',
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f'\nSaved {len(df_out)} rows → {out_path}')


if __name__ == '__main__':
    main()
