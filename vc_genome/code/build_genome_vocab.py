"""
build_genome_vocab.py
=====================
Step 1 of the pure-genome pipeline.

Extracts a raw vocabulary from the 'CuratePhrasesMore/LessComplex' column
across all ~497 images (9 main VisTypes) using spaCy.

For each phrase segment (split on ';') the script extracts:
  - nouns / noun chunks  → candidate object terms
  - verbs (lemmatised)   → candidate predicate stems
  - adjectives           → candidate attribute modifiers
  - full phrases         → candidate attribute / relationship phrases

Output
------
  vc_genome/export/pure_genome_vocab_raw.csv
    term, pos_type, count, freq_pct, example_images (first 3)

Usage
-----
  python code/build_genome_vocab.py
"""
from __future__ import annotations
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import spacy

SCRIPT_DIR = Path(__file__).parent
ROOT       = SCRIPT_DIR.parent.parent           # comment_post_processing/

INPUT_CSV  = ROOT / 'comment_process' / 'ResultsStepByStep_4.0.imageDataCompiled.csv'
EXPORT_DIR = ROOT / 'vc_genome' / 'export'
OUT_CSV    = EXPORT_DIR / 'pure_genome_vocab_raw.csv'

NINE_VISTYPES = ['Area', 'Bar', 'Cont.-ColorPatn', 'Glyph', 'Grid',
                 'Line', 'Node-link', 'Point', 'Text']
PHRASE_COL = 'CuratePhrasesMore/LessComplex'

# spaCy stopwords to drop (in addition to built-in)
EXTRA_STOPWORDS = {
    'image', 'picture', 'graph', 'chart', 'figure', 'visualization',
    'one', 'two', 'other', 'also', 'much', 'many', 'lot', 'lots',
    'thing', 'way', 'type', 'kind', 'look', 'seem', 'use',
}


def clean_segment(seg: str) -> str:
    """Strip sentiment markers, trailing punctuation, and extra whitespace."""
    seg = re.sub(r'\s*\([+\-?]\)\s*', ' ', seg)
    seg = seg.strip(' .,;')
    return seg.lower()


def extract_terms(doc, image_name: str, records: dict):
    """Populate `records` dict from a parsed spaCy doc."""

    # ── noun chunks (phrases like "bar graph", "hard to read") ───────────
    for chunk in doc.noun_chunks:
        term = chunk.lemma_.lower().strip()
        if len(term) < 3 or all(t.is_stop for t in chunk):
            continue
        records['noun_phrase'][term]['count'] += 1
        records['noun_phrase'][term]['images'].add(image_name)

    # ── individual tokens ─────────────────────────────────────────────────
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.is_space:
            continue
        lemma = tok.lemma_.lower().strip()
        if len(lemma) < 3 or lemma in EXTRA_STOPWORDS:
            continue

        if tok.pos_ == 'NOUN':
            records['noun'][lemma]['count'] += 1
            records['noun'][lemma]['images'].add(image_name)
        elif tok.pos_ == 'VERB':
            records['verb'][lemma]['count'] += 1
            records['verb'][lemma]['images'].add(image_name)
        elif tok.pos_ in ('ADJ', 'ADV'):
            records['adj'][lemma]['count'] += 1
            records['adj'][lemma]['images'].add(image_name)


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading spaCy model...')
    try:
        nlp = spacy.load('en_core_web_md')
    except OSError:
        nlp = spacy.load('en_core_web_sm')
    print(f'  Using: {nlp.meta["name"]}')

    df = pd.read_csv(INPUT_CSV)
    df = df[df['VisType'].isin(NINE_VISTYPES)].copy()
    df[PHRASE_COL] = df[PHRASE_COL].fillna('').astype(str)
    print(f'Loaded {len(df)} images from {INPUT_CSV.name}')

    # records[pos_type][term] = {count, images}
    records: dict[str, dict] = {
        'noun':        defaultdict(lambda: {'count': 0, 'images': set()}),
        'verb':        defaultdict(lambda: {'count': 0, 'images': set()}),
        'adj':         defaultdict(lambda: {'count': 0, 'images': set()}),
        'noun_phrase': defaultdict(lambda: {'count': 0, 'images': set()}),
    }

    for _, row in df.iterrows():
        image_name = row['imageName']
        raw = row[PHRASE_COL]
        if not raw.strip():
            continue
        segments = [clean_segment(s) for s in raw.split(';') if s.strip()]
        text = '. '.join(segments)
        doc = nlp(text)
        extract_terms(doc, image_name, records)

    # ── Flatten to rows ───────────────────────────────────────────────────
    total_images = len(df)
    rows = []
    for pos_type, terms in records.items():
        for term, data in terms.items():
            cnt   = data['count']
            imgs  = sorted(data['images'])
            rows.append({
                'term':           term,
                'pos_type':       pos_type,
                'count':          cnt,
                'freq_pct':       round(cnt / total_images * 100, 1),
                'n_images':       len(imgs),
                'example_images': '; '.join(imgs[:3]),
            })

    df_out = (pd.DataFrame(rows)
              .sort_values(['pos_type', 'count'], ascending=[True, False])
              .reset_index(drop=True))

    # Drop very rare (count == 1) single-token entries to reduce noise
    df_out = df_out[~((df_out['count'] == 1) & (df_out['pos_type'] != 'noun_phrase'))]

    df_out.to_csv(OUT_CSV, index=False)
    print(f'\nVocabulary statistics:')
    print(df_out.groupby('pos_type')['term'].count().to_string())
    print(f'\nTotal unique terms: {len(df_out)}')
    print(f'Saved → {OUT_CSV}')


if __name__ == '__main__':
    main()
