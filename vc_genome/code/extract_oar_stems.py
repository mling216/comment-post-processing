"""
extract_oar_stems.py
====================
Adds an 'allStems' column to ResultsStepByStep_4.0.imageDataCompiled.csv.

For each cell in 'CuratePhrasesMore/LessComplex', splits on ';', then uses
spaCy to extract lemmatised tokens that could serve as OAR terms:
  - NOUN / PROPN  →  potential Objects
  - ADJ           →  potential Attributes
  - VERB          →  potential Relationship predicates  (non-aux / non-cop)

Tokens are deduplicated (order-preserving) and joined with '; '.
Rows with no usable phrases get an empty string.
"""
from __future__ import annotations
import re
from pathlib import Path

import pandas as pd
import spacy

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT       = SCRIPT_DIR.parent.parent
INPUT_CSV  = ROOT / 'comment_process' / 'ResultsStepByStep_4.0.imageDataCompiled.csv'
PHRASE_COL = 'CuratePhrasesMore/LessComplex'
OUT_COL    = 'allStems'

OAR_POS = {'NOUN', 'PROPN', 'ADJ', 'ADV', 'ADV', 'VERB'}


def clean_seg(seg: str) -> str:
    seg = re.sub(r'\s*\([+\-?]\)\s*', ' ', seg)   # strip sentiment markers
    return seg.strip(' .,;').lower()


def stems_from_cell(cell: str, nlp) -> str:
    """Return '; '-joined deduplicated OAR stems for one CSV cell."""
    if not isinstance(cell, str) or not cell.strip():
        return ''

    seen: list[str] = []
    seen_set: set[str] = set()

    for raw_seg in cell.split(';'):
        seg = clean_seg(raw_seg)
        if not seg:
            continue
        doc = nlp(seg)
        for tok in doc:
            if tok.is_stop or tok.is_punct or tok.is_space:
                continue
            if tok.pos_ not in OAR_POS:
                continue
            lemma = tok.lemma_.lower().strip()
            if len(lemma) < 3:
                continue
            if lemma not in seen_set:
                seen_set.add(lemma)
                seen.append(lemma)

    return '; '.join(seen)


def main():
    print(f'Loading spaCy model...')
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    print(f'Reading {INPUT_CSV}')
    df = pd.read_csv(INPUT_CSV)

    if PHRASE_COL not in df.columns:
        raise KeyError(f"Column '{PHRASE_COL}' not found in CSV.")

    print(f'Extracting OAR stems from {len(df)} rows...')
    df[OUT_COL] = df[PHRASE_COL].apply(lambda c: stems_from_cell(c, nlp))

    df.to_csv(INPUT_CSV, index=False)
    print(f'Saved → {INPUT_CSV}  ({OUT_COL} column added)')

    # Quick sanity check: print first 5 non-empty results
    sample = df[df[OUT_COL].str.len() > 0][['imageName', PHRASE_COL, OUT_COL]].head(5)
    for _, row in sample.iterrows():
        print(f"\n  {row['imageName']}")
        print(f"    phrases : {row[PHRASE_COL]}")
        print(f"    allStems: {row[OUT_COL]}")


if __name__ == '__main__':
    main()
