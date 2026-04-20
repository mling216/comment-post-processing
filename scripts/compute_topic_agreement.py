"""
Compute agreement between human Topics and LLM topic predictions.
Compares v0_topics and scoring_v4_topics side-by-side.
Metrics: sample-averaged F1, Jaccard, per-topic Cohen's Kappa.

Usage:
    python scripts/compute_topic_agreement.py
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
    n = len(df)
    mat = np.zeros((n, len(ALL_TOPICS)), dtype=int)
    for i, (_, row) in enumerate(df.iterrows()):
        topics = parse_topics(row[col])
        for j, t in enumerate(ALL_TOPICS):
            if t in topics:
                mat[i, j] = 1
    return mat


def compute_metrics(human_mat, llm_mat, label):
    n = human_mat.shape[0]

    # Jaccard
    jaccards = []
    for i in range(n):
        h = set(np.where(human_mat[i])[0])
        l = set(np.where(llm_mat[i])[0])
        jaccards.append(1.0 if len(h | l) == 0 else len(h & l) / len(h | l))

    # F1
    f1_samples = f1_score(human_mat, llm_mat, average='samples')
    f1_macro   = f1_score(human_mat, llm_mat, average='macro')

    # Per-topic Kappa
    kappas = []
    topic_details = []
    for j, t in enumerate(ALL_TOPICS):
        h_col = human_mat[:, j]
        l_col = llm_mat[:, j]
        if len(set(h_col)) == 1 and len(set(l_col)) == 1:
            k = 1.0 if h_col[0] == l_col[0] else 0.0
        else:
            k = cohen_kappa_score(h_col, l_col)
        agree = np.sum(h_col == l_col)
        kappas.append(k)
        topic_details.append((t, k, agree, h_col.sum(), l_col.sum()))

    # Exact & partial match
    exact = sum(1 for i in range(n) if np.array_equal(human_mat[i], llm_mat[i]))
    partial = sum(1 for j in jaccards if j > 0)

    return {
        'label': label,
        'jaccard_mean': np.mean(jaccards),
        'jaccard_median': np.median(jaccards),
        'f1_samples': f1_samples,
        'f1_macro': f1_macro,
        'kappa_macro': np.mean(kappas),
        'exact': exact,
        'partial': partial,
        'n': n,
        'topic_details': topic_details,
    }


def print_metrics(m):
    label = m['label']
    n = m['n']
    print(f'\n{"=" * 60}')
    print(f'  {label}')
    print(f'{"=" * 60}')

    print(f'\n  Jaccard:   mean={m["jaccard_mean"]:.3f}  median={m["jaccard_median"]:.3f}')
    print(f'  Sample F1: {m["f1_samples"]:.3f}   Macro F1: {m["f1_macro"]:.3f}')

    print(f'\n  Per-topic Cohen\'s Kappa:')
    for t, k, agree, h_sum, l_sum in m['topic_details']:
        print(f'    {t:40s}  κ={k:+.3f}  agree={agree}/{n}  human={h_sum:2d}  llm={l_sum:2d}')
    print(f'    {"Macro-average":40s}  κ={m["kappa_macro"]:+.3f}')

    print(f'\n  Exact match:   {m["exact"]}/{n} ({m["exact"]/n*100:.1f}%)')
    print(f'  Partial match: {m["partial"]}/{n} ({m["partial"]/n*100:.1f}%)')


def print_comparison(m_v0, m_v4):
    print(f'\n{"=" * 60}')
    print(f'  SIDE-BY-SIDE COMPARISON')
    print(f'{"=" * 60}')
    print(f'  {"Metric":<25s}  {"v0_topics":>10s}  {"scoring_v4":>10s}  {"Δ (v4−v0)":>10s}')
    print(f'  {"-"*25}  {"-"*10}  {"-"*10}  {"-"*10}')

    rows = [
        ('Sample F1',     m_v0['f1_samples'],    m_v4['f1_samples']),
        ('Macro F1',      m_v0['f1_macro'],      m_v4['f1_macro']),
        ('Jaccard (mean)', m_v0['jaccard_mean'],  m_v4['jaccard_mean']),
        ('Macro κ',       m_v0['kappa_macro'],   m_v4['kappa_macro']),
        ('Exact match %', m_v0['exact']/m_v0['n']*100, m_v4['exact']/m_v4['n']*100),
        ('Partial match %', m_v0['partial']/m_v0['n']*100, m_v4['partial']/m_v4['n']*100),
    ]
    for name, a, b in rows:
        print(f'  {name:<25s}  {a:>10.3f}  {b:>10.3f}  {b-a:>+10.3f}')

    # Per-topic kappa comparison
    print(f'\n  {"Topic":<40s}  {"κ v0":>6s}  {"κ v4":>6s}  {"Δ":>7s}')
    print(f'  {"-"*40}  {"-"*6}  {"-"*6}  {"-"*7}')
    for (t, k0, *_), (_, k4, *_) in zip(m_v0['topic_details'], m_v4['topic_details']):
        print(f'  {t:<40s}  {k0:>+.3f}  {k4:>+.3f}  {k4-k0:>+.3f}')
    print(f'  {"Macro-average":<40s}  {m_v0["kappa_macro"]:>+.3f}  {m_v4["kappa_macro"]:>+.3f}  {m_v4["kappa_macro"]-m_v0["kappa_macro"]:>+.3f}')


def main():
    df = pd.read_csv(RESULTS_CSV)

    human_mat = build_matrix(df, 'Topics')
    v0_mat    = build_matrix(df, 'v0_topics')
    v4_mat    = build_matrix(df, 'scoring_v4_topics')

    m_v0 = compute_metrics(human_mat, v0_mat, 'v0_topics vs Human Topics')
    m_v4 = compute_metrics(human_mat, v4_mat, 'scoring_v4_topics vs Human Topics')

    print_metrics(m_v0)
    print_metrics(m_v4)
    print_comparison(m_v0, m_v4)

    # v0 vs v4 (inter-method agreement)
    m_inter = compute_metrics(v0_mat, v4_mat, 'v0_topics vs scoring_v4_topics (inter-method)')
    print_metrics(m_inter)


if __name__ == '__main__':
    main()
