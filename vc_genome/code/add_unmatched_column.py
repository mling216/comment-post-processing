"""
add_unmatched_column.py
=======================
Adds an 'unmatched_stems' column to ResultsStepByStep_4.0.imageDataCompiled.csv.

For each row, identifies allStems tokens that were NOT matched into any of the
visual_objects / visual_attributes / visual_predicates columns, then annotates
each with the reason it was excluded from the dict:

  not in raw vocab   – never extracted as a standalone token during vocab building;
                       sub-labelled by frequency in allStems:
                         rare (N image/s)   – appeared in N rows (N ≤ 4)
                         filtered (N images) – appeared in N rows but excluded
                                              as too generic/non-visual
  in vocab, excluded – appeared in pure_genome_vocab_raw.csv but was not promoted
                       to the curated dict (low frequency or curation judgment)
  unclustered        – in pure_genome_dict.json's 'unclustered' section (was
                       extracted but left un-categorised)

Output format:  "term1 (reason); term2 (reason); ..."
Empty string if all stems were matched.
"""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT       = SCRIPT_DIR.parent.parent
INPUT_CSV  = ROOT / 'comment_process' / 'ResultsStepByStep_4.0.imageDataCompiled.csv'
DICT_JSON  = ROOT / 'vc_genome' / 'export' / 'pure_genome_dict.json'
VOCAB_CSV  = ROOT / 'vc_genome' / 'export' / 'pure_genome_vocab_raw.csv'

STEMS_COL    = 'allStems'
VISUAL_COLS  = ['visual_objects', 'visual_attributes', 'visual_predicates']
OUT_COL      = 'unmatched_stems'
SECTIONS     = ['visual_objects', 'visual_attributes', 'visual_predicates']

# ── helpers ───────────────────────────────────────────────────────────────────
def build_curated_single_terms(data: dict) -> tuple[set[str], dict[str, str]]:
    """
    Returns (set of all single-word curated terms,
             dict of term → 'section.subcat' label for the first match).
    """
    result: set[str] = set()
    section_map: dict[str, str] = {}
    for section in SECTIONS:
        for subcat, terms in data[section].items():
            for t in terms:
                t = t.lower().strip()
                if ' ' not in t:
                    result.add(t)
                    if t not in section_map:
                        section_map[t] = f'{section}.{subcat}'
    return result, section_map


def build_unclustered_terms(data: dict) -> set[str]:
    """All terms in the unclustered section (any pos_type)."""
    result = set()
    unc = data.get('unclustered', {})
    for terms in unc.values():
        for t in terms:
            result.add(t.lower().strip())
    return result


def extract_matched_single_stems(row: pd.Series) -> set[str]:
    """
    Parse the three visual_* columns to find which single-word stems were matched.
    Each column has entries like  "bar (mark); bar graph (chart_type)"
    We extract the term part (before the last space+parenthetical) and keep
    only single-word ones, since only those come from allStems matching.
    """
    matched = set()
    pattern = re.compile(r'^(.+?)\s+\([^)]+\)$')
    for col in VISUAL_COLS:
        cell = row.get(col, '')
        if not isinstance(cell, str) or not cell.strip():
            continue
        for entry in cell.split(';'):
            entry = entry.strip()
            m = pattern.match(entry)
            if m:
                term = m.group(1).lower().strip()
                if ' ' not in term:        # single-word only
                    matched.add(term)
    return matched


def classify_reason(term: str,
                    curated: set[str],
                    unclustered: set[str],
                    raw_vocab: set[str],
                    term_section_map: dict[str, str],
                    stems_freq: dict[str, int]) -> str:
    # In the curated dict — shouldn't normally appear as unmatched.
    if term in curated:
        return 'in curated dict'
    if term in unclustered:
        return 'unclustered'
    if term in raw_vocab:
        return 'in vocab, excluded'
    # Not in raw vocab: sub-label by how often it appears in allStems.
    n = stems_freq.get(term, 0)
    if n <= 4:
        img_word = 'image' if n == 1 else 'images'
        return f'rare, {n} {img_word}'
    return f'filtered, {n} images'


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f'Loading dict: {DICT_JSON}')
    with open(DICT_JSON, encoding='utf-8') as f:
        data = json.load(f)

    curated_single, term_section_map = build_curated_single_terms(data)
    unclustered    = build_unclustered_terms(data)
    print(f'  curated single-word terms: {len(curated_single)}')
    print(f'  unclustered terms:         {len(unclustered)}')

    print(f'Loading raw vocab: {VOCAB_CSV}')
    vocab_df   = pd.read_csv(VOCAB_CSV)
    raw_vocab  = set(vocab_df['term'].str.lower().str.strip())
    print(f'  raw vocab size: {len(raw_vocab)}')

    print(f'Reading CSV: {INPUT_CSV}')
    df = pd.read_csv(INPUT_CSV)

    if STEMS_COL not in df.columns:
        raise KeyError(f"Column '{STEMS_COL}' not found — run extract_oar_stems.py first.")
    for col in VISUAL_COLS:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found — run map_dict_to_phrases.py first.")

    # Pre-compute how many rows each stem appears in (used for 'rare/filtered' labels).
    stems_freq: Counter = Counter()
    for cell in df[STEMS_COL]:
        if isinstance(cell, str) and cell.strip():
            for s in cell.split(';'):
                s = s.strip()
                if s:
                    stems_freq[s] += 1

    unmatched_col = []
    for _, row in df.iterrows():
        stems_raw = row.get(STEMS_COL, '')
        if not isinstance(stems_raw, str) or not stems_raw.strip():
            unmatched_col.append('')
            continue

        all_stems   = [s.strip() for s in stems_raw.split(';') if s.strip()]
        matched     = extract_matched_single_stems(row)
        unmatched   = [s for s in all_stems if s not in matched]

        if not unmatched:
            unmatched_col.append('')
            continue

        parts = []
        for term in unmatched:
            reason = classify_reason(term, curated_single, unclustered, raw_vocab, term_section_map, stems_freq)
            parts.append(f'{term} ({reason})')
        unmatched_col.append('; '.join(parts))

    df[OUT_COL] = unmatched_col

    df.to_csv(INPUT_CSV, index=False)
    print(f'Saved with new column "{OUT_COL}" → {INPUT_CSV}')

    # ── quick stats ────────────────────────────────────────────────────────
    total_rows     = len(df)
    rows_with_any  = df[OUT_COL].str.strip().astype(bool).sum()
    all_reasons    = []
    for cell in df[OUT_COL]:
        if not isinstance(cell, str) or not cell.strip():
            continue
        for entry in cell.split(';'):
            entry = entry.strip()
            m = re.search(r'\((.+?)\)$', entry)
            if m:
                all_reasons.append(m.group(1))

    counts = Counter(all_reasons)
    print(f'\nRows with any unmatched stems : {rows_with_any}/{total_rows}')
    print('Reason breakdown (across all rows × terms):')
    for reason, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f'  {reason:<22} {n}')


if __name__ == '__main__':
    main()
