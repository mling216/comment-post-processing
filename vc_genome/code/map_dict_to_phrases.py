"""
map_dict_to_phrases.py
======================
For each image row in ResultsStepByStep_4.0.imageDataCompiled.csv, finds
which terms from pure_genome_dict.json are mentioned in the
'CuratePhrasesMore/LessComplex' phrases and adds three columns:

  visual_objects    – "term (subcategory); ..."
  visual_attributes – "term (subcategory); ..."
  visual_predicates – "term (subcategory); ..."

Matching strategy
-----------------
- Multi-word dict terms  → regex word-boundary search in lowercased raw phrase
- Single-word dict terms → exact match against lemmatised allStems tokens
  (so "axes" → "axis" is captured via the stem, "bars" → "bar", etc.)

Terms that appear in multiple subcategories are listed once per (section, subcat)
pair.  The 'unclustered' section of the dict is intentionally skipped.
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT       = SCRIPT_DIR.parent.parent
INPUT_CSV  = ROOT / 'comment_process' / 'ResultsStepByStep_4.0.imageDataCompiled.csv'
DICT_JSON  = ROOT / 'vc_genome' / 'export' / 'pure_genome_dict.json'

PHRASE_COL = 'CuratePhrasesMore/LessComplex'
STEMS_COL  = 'allStems'
SECTIONS   = ['visual_objects', 'visual_attributes', 'visual_predicates']

# ── build lookup ──────────────────────────────────────────────────────────────
def build_lookup(data: dict) -> tuple[dict, dict]:
    """
    Returns:
      single_lookup  : lemma → list of (section, subcat)
      multi_lookup   : phrase → list of (section, subcat)   (sorted longest first)
    """
    single: dict[str, list] = defaultdict(list)
    multi:  dict[str, list] = defaultdict(list)

    for section in SECTIONS:
        for subcat, terms in data[section].items():
            for term in terms:
                key = term.lower().strip()
                if ' ' in key:
                    multi[key].append((section, subcat))
                else:
                    single[key].append((section, subcat))

    # Sort multi-word keys longest first so longer matches take priority
    multi_sorted = dict(sorted(multi.items(), key=lambda x: len(x[0]), reverse=True))
    return single, multi_sorted


# ── per-row matching ──────────────────────────────────────────────────────────
def match_row(phrase_raw, stems_raw,
              single_lookup: dict, multi_lookup: dict) -> dict[str, list]:
    """
    Returns { section: [(term, subcat), ...] } with no duplicates per term.
    """
    phrase_lower = phrase_raw.lower() if isinstance(phrase_raw, str) else ''
    stems: list[str] = (
        [s.strip() for s in stems_raw.split(';') if s.strip()]
        if isinstance(stems_raw, str) else []
    )
    stems_set = set(stems)

    # Track which terms were already matched (avoid double-counting)
    matched_terms: set[str] = set()
    result: dict[str, list] = defaultdict(list)

    # 1. Multi-word terms: word-boundary regex on raw phrase
    for term, entries in multi_lookup.items():
        if term in matched_terms:
            continue
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, phrase_lower):
            for section, subcat in entries:
                result[section].append((term, subcat))
            matched_terms.add(term)

    # 2. Single-word terms: exact match against allStems
    for stem in stems:
        if stem in matched_terms:
            continue
        if stem in single_lookup:
            for section, subcat in single_lookup[stem]:
                result[section].append((stem, subcat))
            matched_terms.add(stem)

    return result


def format_column(matches: list[tuple[str, str]]) -> str:
    """[(term, subcat), ...] → 'term (subcat); term2 (subcat2)'"""
    # Deduplicate while preserving order
    seen = set()
    parts = []
    for term, subcat in matches:
        key = (term, subcat)
        if key not in seen:
            seen.add(key)
            parts.append(f'{term} ({subcat})')
    return '; '.join(parts)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f'Loading dict: {DICT_JSON}')
    with open(DICT_JSON, encoding='utf-8') as f:
        data = json.load(f)

    single_lookup, multi_lookup = build_lookup(data)
    print(f'  single-word terms: {len(single_lookup)}')
    print(f'  multi-word terms:  {len(multi_lookup)}')

    print(f'Reading {INPUT_CSV}')
    df = pd.read_csv(INPUT_CSV)

    # Ensure allStems exists
    if STEMS_COL not in df.columns:
        raise KeyError(f"Column '{STEMS_COL}' not found — run extract_oar_stems.py first.")

    rows_obj, rows_attr, rows_pred = [], [], []
    for _, row in df.iterrows():
        matches = match_row(row.get(PHRASE_COL), row.get(STEMS_COL),
                            single_lookup, multi_lookup)
        rows_obj.append( format_column(matches.get('visual_objects',    [])))
        rows_attr.append(format_column(matches.get('visual_attributes', [])))
        rows_pred.append(format_column(matches.get('visual_predicates', [])))

    df['visual_objects']    = rows_obj
    df['visual_attributes'] = rows_attr
    df['visual_predicates'] = rows_pred

    df.to_csv(INPUT_CSV, index=False)
    print(f'Saved → {INPUT_CSV}')

    # Spot-check
    sample = df[df['visual_objects'].str.len() > 0].head(5)
    for _, row in sample.iterrows():
        print(f"\n  {row['imageName']}")
        print(f"    phrases   : {row[PHRASE_COL]}")
        print(f"    objects   : {row['visual_objects']}")
        print(f"    attributes: {row['visual_attributes']}")
        print(f"    predicates: {row['visual_predicates']}")


if __name__ == '__main__':
    main()
