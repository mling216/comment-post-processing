"""
Extended alignment metrics (Opus 4.7 recommendations)
======================================================
1. Bootstrap 95% BCa confidence intervals on CCC and Spearman ρ
   for every VC-scoring variant on the 63-image, 510-image, and
   1800-image corpora.
2. Jaccard similarity and cardinality-stratified macro-F1 for topic
   selection on the 63-image and 510-image corpora.

Output:
  results/alignment_vc_bootstrap.csv
  results/alignment_topic_jaccard.csv
  results/alignment_topic_card_stratified.csv
  (all results also printed to stdout)

Usage:
    python scripts/alignment_extended_metrics.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT    = Path(__file__).parent.parent
RESULTS = ROOT / 'results'
ANCHORS = {'VisC.503.6.png', 'InfoVisJ.619.17.png', 'InfoVisJ.1149.6(1).png'}

sys.path.insert(0, str(Path(__file__).parent))
from topic_select_preview import DIMS, TOPIC_TO_DIM

# ─── metric helpers ──────────────────────────────────────────────────────────

def ccc(y: np.ndarray, yh: np.ndarray) -> float:
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    s = np.cov(y, yh, ddof=0)[0, 1]
    return float((2 * s) / (y.var() + yh.var() + (y.mean() - yh.mean()) ** 2))


def spearman(y: np.ndarray, yh: np.ndarray) -> float:
    return float(stats.spearmanr(y, yh)[0])


# ─── BCa bootstrap ───────────────────────────────────────────────────────────

def bca_ci(y: np.ndarray, yh: np.ndarray, stat_fn,
           n_boot: int = 2000, alpha: float = 0.05,
           rng: np.random.Generator | None = None) -> tuple[float, float]:
    """
    Bias-corrected and accelerated (BCa) bootstrap CI.
    Returns (lower, upper) for confidence level 1-alpha.
    For large n (>200), uses percentile bootstrap to keep runtime reasonable.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(y)
    observed = stat_fn(y, yh)

    # bootstrap distribution
    idx_boot = rng.integers(0, n, size=(n_boot, n))
    boot = np.array([stat_fn(y[idx], yh[idx]) for idx in idx_boot])

    if n > 200:
        # For large n, BCa jackknife is slow; fall back to percentile bootstrap
        lo = float(np.percentile(boot, 100 * alpha / 2))
        hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
        return lo, hi

    # bias correction z0
    z0 = stats.norm.ppf(np.mean(boot < observed))

    # acceleration a (jackknife) — only for small n
    jack = np.array([
        stat_fn(np.delete(y, i), np.delete(yh, i))
        for i in range(n)
    ])
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    den = 6 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    # adjusted quantiles
    z_alpha   = stats.norm.ppf(alpha / 2)
    z_1alpha  = stats.norm.ppf(1 - alpha / 2)

    def adj_quantile(z: float) -> float:
        num_z = z0 + z
        pct = stats.norm.cdf(z0 + num_z / (1 - a * num_z))
        pct = np.clip(pct, 0, 1)
        return float(np.percentile(boot, 100 * pct))

    lo = adj_quantile(z_alpha)
    hi = adj_quantile(z_1alpha)
    return lo, hi


# ─── VC bootstrap CIs ────────────────────────────────────────────────────────

def load_vc_pairs(pred_csv: Path, gt_df: pd.DataFrame,
                  gt_col: str = 'NormalizedVC',
                  exclude_anchors: bool = True) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(pred_csv).rename(columns={'filename': 'imageName'})
    m = gt_df.merge(df, on='imageName')
    if exclude_anchors:
        m = m[~m['imageName'].isin(ANCHORS)]
    return m[gt_col].astype(float).values, m['vc_score'].astype(float).values


def run_vc_bootstrap(variants: list[tuple[str, Path, pd.DataFrame, str]],
                     n_boot: int = 2000) -> pd.DataFrame:
    """
    variants: list of (label, pred_csv, gt_df, gt_col)
    Returns a DataFrame with one row per variant.
    """
    rng = np.random.default_rng(42)
    rows = []
    for label, pred_csv, gt_df, gt_col in variants:
        if not pred_csv.exists():
            print(f'  SKIP {label}: {pred_csv} not found')
            continue
        try:
            y, yh = load_vc_pairs(pred_csv, gt_df, gt_col)
        except Exception as e:
            print(f'  ERROR {label}: {e}')
            continue

        c_obs   = ccc(y, yh)
        rho_obs = spearman(y, yh)

        c_lo, c_hi     = bca_ci(y, yh, ccc,     n_boot=n_boot, rng=rng)
        rho_lo, rho_hi = bca_ci(y, yh, spearman, n_boot=n_boot, rng=rng)

        rows.append(dict(
            variant=label, n=len(y),
            ccc=round(c_obs, 4),
            ccc_lo=round(c_lo, 4),  ccc_hi=round(c_hi, 4),
            ccc_ci=f'[{c_lo:.4f}, {c_hi:.4f}]',
            spearman_rho=round(rho_obs, 4),
            rho_lo=round(rho_lo, 4), rho_hi=round(rho_hi, 4),
            rho_ci=f'[{rho_lo:.4f}, {rho_hi:.4f}]',
        ))
        print(f'  {label:<22} n={len(y):>4}  '
              f'CCC={c_obs:.4f} 95%CI{c_lo:.4f},{c_hi:.4f}  '
              f'rho={rho_obs:.4f} 95%CI{rho_lo:.4f},{rho_hi:.4f}')
    return pd.DataFrame(rows)


# ─── Topic selection: Jaccard + cardinality-stratified F1 ────────────────────

def jaccard(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    union = pred | gold
    return len(pred & gold) / len(union) if union else 0.0


def f1_set(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def build_topic_gt_63() -> pd.DataFrame:
    """Return per-image GT from analysis_63.csv (already computed)."""
    df = pd.read_csv(RESULTS / 'probe_prominent_63/analysis_63.csv')
    df = df.rename(columns={'gt': 'human_topics', 'vistype': 'VisType'})
    df['n_human'] = df['human_topics'].apply(
        lambda x: len([t for t in str(x).split(';') if t.strip()])
    )
    return df


def build_topic_gt_510() -> pd.DataFrame:
    """Build per-image human GT for the 510-image set from compiled phrases."""
    compiled = pd.read_csv(
        ROOT / 'phrase_reduction_v2/image_compiled_phrases.csv'
    )[['imageName', 'Topics', 'VisType', 'NormalizedVC']]
    inp = pd.read_csv(RESULTS / 'vc_api_510_v0_tw_input.csv')[['imageName', 'VisType']]

    rows = []
    for fn in inp['imageName']:
        if fn in ANCHORS:
            continue
        m = compiled[compiled['imageName'] == fn]
        if len(m) == 0:
            continue
        topics_raw = str(m.iloc[0]['Topics'])
        dims = []
        for t in [x.strip() for x in topics_raw.split(';') if x.strip()]:
            if t in TOPIC_TO_DIM:
                d = TOPIC_TO_DIM[t]
                if d not in dims:
                    dims.append(d)
        vt = m.iloc[0]['VisType'] if 'VisType' in m.columns else inp.loc[inp['imageName'] == fn, 'VisType'].iloc[0]
        rows.append({
            'filename': fn,
            'VisType': vt,
            'human_topics': ';'.join(dims),
            'n_human': len(dims),
        })
    return pd.DataFrame(rows)


def topic_metrics_per_image(gt_df: pd.DataFrame,
                             pred_col: str,
                             pred_df: pd.DataFrame | None = None,
                             pred_csv: Path | None = None,
                             pred_field: str = 'top3_topics') -> pd.DataFrame:
    """
    Compute per-image Jaccard, F1, precision, recall.
    Either pass pred_df (already loaded) or pred_csv (path to load).
    pred_col is a label string for the result column prefix.
    """
    if pred_df is None:
        assert pred_csv is not None
        pred_df = pd.read_csv(pred_csv)
        if 'filename' in pred_df.columns:
            pred_df = pred_df.rename(columns={'filename': 'filename'})

    if 'filename' not in gt_df.columns:
        gt_df = gt_df.rename(columns={'imageName': 'filename'})

    join_col = 'filename'
    m = gt_df[[join_col, 'human_topics', 'n_human', 'VisType']].merge(
        pred_df[[join_col, pred_field]], on=join_col, how='inner'
    )

    rows = []
    for _, r in m.iterrows():
        gold = set(t.strip() for t in str(r['human_topics']).split(';') if t.strip())
        pred = set(t.strip() for t in str(r[pred_field]).split(';') if t.strip())
        tp = len(pred & gold)
        p  = tp / len(pred) if pred else 0.0
        rec= tp / len(gold) if gold else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        rows.append({
            join_col: r[join_col],
            'VisType': r['VisType'],
            'n_human': r['n_human'],
            'f1': f1,
            'precision': p,
            'recall': rec,
            'jaccard': jaccard(pred, gold),
        })
    return pd.DataFrame(rows)


def summarise_topic(per_img: pd.DataFrame, label: str) -> dict:
    return dict(
        variant=label,
        n=len(per_img),
        macro_f1=round(per_img['f1'].mean(), 4),
        macro_jaccard=round(per_img['jaccard'].mean(), 4),
        macro_precision=round(per_img['precision'].mean(), 4),
        macro_recall=round(per_img['recall'].mean(), 4),
    )


def stratified_f1(per_img: pd.DataFrame, label: str) -> list[dict]:
    """Return per-cardinality-bin macro-F1 row."""
    bins = [(1, 1), (2, 2), (3, 3), (4, 7)]
    rows = []
    for lo, hi in bins:
        sub = per_img[(per_img['n_human'] >= lo) & (per_img['n_human'] <= hi)]
        rows.append(dict(
            variant=label,
            cardinality=f'{lo}' if lo == hi else f'{lo}–{hi}',
            n=len(sub),
            macro_f1=round(sub['f1'].mean(), 4) if len(sub) > 0 else float('nan'),
            macro_jaccard=round(sub['jaccard'].mean(), 4) if len(sub) > 0 else float('nan'),
            macro_precision=round(sub['precision'].mean(), 4) if len(sub) > 0 else float('nan'),
            macro_recall=round(sub['recall'].mean(), 4) if len(sub) > 0 else float('nan'),
        ))
    return rows


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    print('\n' + '=' * 65)
    print('  PART 1: Bootstrap 95% BCa CIs on CCC and Spearman rho')
    print('=' * 65)

    # ── 63-image GT ──
    gt66 = pd.read_csv(ROOT / 'Claude_vc_prediction/gt_all_66.csv')

    # ── 510-image GT ──
    compiled = pd.read_csv(
        ROOT / 'phrase_reduction_v2/image_compiled_phrases.csv'
    )[['imageName', 'NormalizedVC', 'VisType']].drop_duplicates('imageName')

    # ── 1800-image GT ──
    gt1800 = pd.read_csv(RESULTS / 'vc_api_1800_v0_tw_input.csv').rename(
        columns={'imageName': 'imageName', 'NormalizedVC': 'NormalizedVC'}
    )

    # Build variant list  (label, pred_csv, gt_df, gt_col)
    vc_variants = [
        # 63-image balanced set
        ('V0 (63)',      RESULTS/'vc_api_63_vanilla/vc_scores.csv',        gt66,       'NormalizedVC'),
        ('V0+T (63)',    RESULTS/'vc_api_63_vanilla_t/vc_scores.csv',       gt66,       'NormalizedVC'),
        ('V0+A (63)',    RESULTS/'vc_api_63_vanilla_a/vc_scores.csv',       gt66,       'NormalizedVC'),
        ('V0+TW (63)',   RESULTS/'vc_api_63_vanilla_tw_dyn/vc_scores.csv',  gt66,       'NormalizedVC'),
        ('V0+TWA (63)',  RESULTS/'vc_api_63_vanilla_twa/vc_scores.csv',     gt66,       'NormalizedVC'),
        # 510-image full set
        ('V0+TW (510)',     RESULTS/'vc_api_510_v0_tw_dyn/vc_scores.csv',         compiled, 'NormalizedVC'),
        ('V0+TW-det (510)', RESULTS/'vc_api_510_v0_twdyn_det_t0/vc_scores.csv',   compiled, 'NormalizedVC'),
        # 1800-image corpus
        ('V0 (1800)',    RESULTS/'vc_api_1800_v0/vc_scores.csv',           gt1800,     'NormalizedVC'),
        ('V0+TW (1800)', RESULTS/'vc_api_1800_v0_tw_dyn/vc_scores.csv',    gt1800,     'NormalizedVC'),
    ]

    print(f'\n  BCa bootstrap with n_boot=2000, alpha=0.05 ...\n')
    vc_df = run_vc_bootstrap(vc_variants, n_boot=2000)

    out_vc = RESULTS / 'alignment_vc_bootstrap.csv'
    vc_df.to_csv(out_vc, index=False)
    print(f'\n  Saved → {out_vc}')

    # ── TOST equivalence delta table ──
    print('\n  Pairwise dCCC (V0+T vs V0+TW on 63-image, for TOST reference):')
    sub63 = vc_df[vc_df['variant'].isin(['V0+T (63)', 'V0+TW (63)'])]
    if len(sub63) == 2:
        d = abs(sub63.iloc[0]['ccc'] - sub63.iloc[1]['ccc'])
        print(f'    |dCCC| = {d:.4f}  (CI overlap check: '
              f'V0+T {sub63.iloc[0]["ccc_ci"]}, V0+TW {sub63.iloc[1]["ccc_ci"]})')

    # ─────────────────────────────────────────────────────────────────────────
    print('\n\n' + '=' * 65)
    print('  PART 2: Topic Selection — Jaccard + Cardinality-Stratified F1')
    print('=' * 65)

    # ── 63-image ──
    print('\n  [63-image balanced set]\n')
    gt63_topic = build_topic_gt_63()

    topic_summary_rows = []
    topic_strat_rows   = []

    # V0+T (top3) — prediction in analysis_63.csv column vot_top3
    for variant_label, pred_field, pred_df_src in [
        ('V0+T top3 (63)',   'vot_top3',  gt63_topic),
        ('V0+TW top3 (63)',  'votw_top3', gt63_topic),
        ('Prominent (63)',   'prominent', gt63_topic),
    ]:
        per_img = topic_metrics_per_image(
            gt63_topic, variant_label,
            pred_df=pred_df_src.rename(columns={'filename': 'filename'}),
            pred_field=pred_field,
        )
        s = summarise_topic(per_img, variant_label)
        topic_summary_rows.append(s)
        topic_strat_rows.extend(stratified_f1(per_img, variant_label))

        print(f'  {variant_label:<22} n={s["n"]:>3}  '
              f'F1={s["macro_f1"]:.4f}  Jaccard={s["macro_jaccard"]:.4f}  '
              f'P={s["macro_precision"]:.4f}  R={s["macro_recall"]:.4f}')

    # ── cardinality breakdown 63 ──
    print('\n  Cardinality-stratified F1 (63-image):')
    _strat63 = [r for r in topic_strat_rows if '(63)' in r['variant']]
    strat_df63 = pd.DataFrame(_strat63)
    if not strat_df63.empty:
        pivot = strat_df63.pivot(index='cardinality', columns='variant', values='macro_f1')
        counts = strat_df63[strat_df63['variant'] == _strat63[0]['variant']][['cardinality', 'n']].set_index('cardinality')
        pivot.insert(0, 'n', counts['n'])
        print(pivot.to_string())

    # ── 510-image ──
    print('\n\n  [510-image full corpus]\n')
    gt510_topic = build_topic_gt_510()
    print(f'  510-image GT: {len(gt510_topic)} images, '
          f'median n_human={gt510_topic.n_human.median():.0f}, '
          f'mean={gt510_topic.n_human.mean():.2f}, '
          f'range {gt510_topic.n_human.min()}–{gt510_topic.n_human.max()}')

    for variant_label, csv_rel in [
        ('V0+T top3 (510)',  'vc_api_510_topicsel_v0_t/vc_scores.csv'),
        ('V0+TW top3 (510)', 'vc_api_510_topicsel_v0_tw/vc_scores.csv'),
    ]:
        csv_path = RESULTS / csv_rel
        if not csv_path.exists():
            print(f'  SKIP {variant_label}: {csv_path} not found')
            continue
        per_img = topic_metrics_per_image(
            gt510_topic, variant_label,
            pred_csv=csv_path,
            pred_field='top3_topics',
        )
        s = summarise_topic(per_img, variant_label)
        topic_summary_rows.append(s)
        topic_strat_rows.extend(stratified_f1(per_img, variant_label))

        print(f'  {variant_label:<24} n={s["n"]:>3}  '
              f'F1={s["macro_f1"]:.4f}  Jaccard={s["macro_jaccard"]:.4f}  '
              f'P={s["macro_precision"]:.4f}  R={s["macro_recall"]:.4f}')

    # prominent 510
    for variant_label, subdir in [
        ('Prominent (510, opus)',   'probe_prominent_510_opus_4_6'),
        ('Prominent (510, sonnet)', 'probe_prominent_510_sonnet-4-5'),
    ]:
        csv_path = RESULTS / subdir / 'prominent_results.csv'
        if not csv_path.exists():
            print(f'  SKIP {variant_label}: {csv_path} not found')
            continue
        per_img = topic_metrics_per_image(
            gt510_topic, variant_label,
            pred_csv=csv_path,
            pred_field='prominent_topics',
        )
        s = summarise_topic(per_img, variant_label)
        topic_summary_rows.append(s)
        topic_strat_rows.extend(stratified_f1(per_img, variant_label))
        print(f'  {variant_label:<28} n={s["n"]:>3}  '
              f'F1={s["macro_f1"]:.4f}  Jaccard={s["macro_jaccard"]:.4f}  '
              f'P={s["macro_precision"]:.4f}  R={s["macro_recall"]:.4f}')

    # ── cardinality breakdown 510 ──
    print('\n  Cardinality-stratified F1 (510-image):')
    _strat510 = [r for r in topic_strat_rows if '(510' in r['variant']]
    strat_df510 = pd.DataFrame(_strat510)
    if not strat_df510.empty:
        pivot510 = strat_df510.pivot(index='cardinality', columns='variant', values='macro_f1')
        counts510 = strat_df510[strat_df510['variant'] == _strat510[0]['variant']][['cardinality', 'n']].set_index('cardinality')
        pivot510.insert(0, 'n', counts510['n'])
        print(pivot510.to_string())

    # ── save outputs ──
    out_jaccard = RESULTS / 'alignment_topic_jaccard.csv'
    out_strat   = RESULTS / 'alignment_topic_card_stratified.csv'
    pd.DataFrame(topic_summary_rows).to_csv(out_jaccard, index=False)
    pd.DataFrame(topic_strat_rows).to_csv(out_strat,   index=False)
    print(f'\n  Saved → {out_jaccard}')
    print(f'  Saved → {out_strat}')

    print('\nDone.\n')


if __name__ == '__main__':
    main()
