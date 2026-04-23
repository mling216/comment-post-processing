"""
Per-VisType topic agreement: v0 vs Human, v4 vs Human, v0 vs v4.
Pools all 7 topic decisions within each VisType (n=3 images → 21 binary decisions).

Usage:
    python scripts/compute_topic_agreement_by_vistype.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

RESULTS_CSV = 'vc_genome_output_full/ablation/subset_27_with_results_v4.csv'

ALL_TOPICS = [
    'Data Density / Image Clutter',
    'Visual Encoding Clarity',
    'Semantics / Text Legibility',
    'Schema',
    'Color, Symbol, and Texture Details',
    'Aesthetics Uncertainty',
    'Immediacy / Cognitive Load',
]


def parse_topics(s):
    if pd.isna(s):
        return set()
    return set(t.strip() for t in str(s).split(';'))


def build_matrix(df, col):
    mat = np.zeros((len(df), len(ALL_TOPICS)), dtype=int)
    for i, (_, row) in enumerate(df.iterrows()):
        topics = parse_topics(row[col])
        for j, t in enumerate(ALL_TOPICS):
            if t in topics:
                mat[i, j] = 1
    return mat


def safe_kappa(a, b):
    if len(set(a)) == 1 and len(set(b)) == 1:
        return 1.0 if a[0] == b[0] else 0.0
    return cohen_kappa_score(a, b)


def main():
    df = pd.read_csv(RESULTS_CSV)

    # ── Per-VisType pooled kappa ──
    header = (
        f"{'VisType':<18s}  {'n':>2s}  "
        f"{'v0↔H F1':>8s}  {'v4↔H F1':>8s}  "
        f"{'v0↔H κ':>7s}  {'v4↔H κ':>7s}  {'v0↔v4 κ':>8s}"
    )
    print(header)
    print('-' * len(header))

    rows = []
    for vt in sorted(df['VisType'].unique()):
        sub = df[df['VisType'] == vt].reset_index(drop=True)
        h  = build_matrix(sub, 'Topics')
        v0 = build_matrix(sub, 'v0_topics')
        v4 = build_matrix(sub, 'scoring_v4_topics')

        # Pooled kappa: flatten all (image × topic) pairs → 21 binary decisions
        h_flat  = h.flatten()
        v0_flat = v0.flatten()
        v4_flat = v4.flatten()

        k_v0h  = safe_kappa(h_flat, v0_flat)
        k_v4h  = safe_kappa(h_flat, v4_flat)
        k_v0v4 = safe_kappa(v0_flat, v4_flat)

        f1_v0 = f1_score(h, v0, average='samples')
        f1_v4 = f1_score(h, v4, average='samples')

        print(
            f"{vt:<18s}  {len(sub):>2d}  "
            f"{f1_v0:>8.3f}  {f1_v4:>8.3f}  "
            f"{k_v0h:>+7.3f}  {k_v4h:>+7.3f}  {k_v0v4:>+8.3f}"
        )
        rows.append({
            'VisType': vt, 'n': len(sub),
            'v0_human_F1': round(f1_v0, 3), 'v4_human_F1': round(f1_v4, 3),
            'v0_human_kappa': round(k_v0h, 3), 'v4_human_kappa': round(k_v4h, 3),
            'v0_v4_kappa': round(k_v0v4, 3),
        })

    # ── Overall (all 27) ──
    h_all  = build_matrix(df, 'Topics')
    v0_all = build_matrix(df, 'v0_topics')
    v4_all = build_matrix(df, 'scoring_v4_topics')

    k_v0h_all  = safe_kappa(h_all.flatten(), v0_all.flatten())
    k_v4h_all  = safe_kappa(h_all.flatten(), v4_all.flatten())
    k_v0v4_all = safe_kappa(v0_all.flatten(), v4_all.flatten())
    f1_v0_all  = f1_score(h_all, v0_all, average='samples')
    f1_v4_all  = f1_score(h_all, v4_all, average='samples')

    print('-' * len(header))
    print(
        f"{'ALL':<18s}  {len(df):>2d}  "
        f"{f1_v0_all:>8.3f}  {f1_v4_all:>8.3f}  "
        f"{k_v0h_all:>+7.3f}  {k_v4h_all:>+7.3f}  {k_v0v4_all:>+8.3f}"
    )
    rows.append({
        'VisType': 'ALL', 'n': len(df),
        'v0_human_F1': round(f1_v0_all, 3), 'v4_human_F1': round(f1_v4_all, 3),
        'v0_human_kappa': round(k_v0h_all, 3), 'v4_human_kappa': round(k_v4h_all, 3),
        'v0_v4_kappa': round(k_v0v4_all, 3),
    })

    # Save CSV
    out_csv = 'vc_genome_output_full/ablation/topic_agreement_by_vistype.csv'
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'\nSaved to {out_csv}')


if __name__ == '__main__':
    main()
