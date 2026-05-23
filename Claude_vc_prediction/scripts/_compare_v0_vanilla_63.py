"""Compare V0 vs Vanilla zero-shot on the 63-image pilot set.
Covers both VC-only (V0) and topic-weighted-dynamic (V0+TW-dyn) variants."""
import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_absolute_error

ROOT    = Path(__file__).parent.parent
ANCHORS = {'VisC.503.6.png', 'InfoVisJ.619.17.png', 'InfoVisJ.1149.6(1).png'}

gt = pd.read_csv(ROOT / 'Claude_vc_prediction/gt_all_66.csv')

def ccc(y, yh):
    y, yh = np.array(y, dtype=float), np.array(yh, dtype=float)
    s = np.cov(y, yh, ddof=0)[0, 1]
    return (2 * s) / (y.var() + yh.var() + (y.mean() - yh.mean()) ** 2)

def metrics(path, label):
    df = pd.read_csv(path).rename(columns={'filename': 'imageName'})
    m  = gt.merge(df, on='imageName')
    m  = m[~m['imageName'].isin(ANCHORS)]
    y  = m['NormalizedVC'].values.astype(float)
    yh = m['vc_score'].astype(float).values
    r,   _ = stats.pearsonr(y, yh)
    rho, _ = stats.spearmanr(y, yh)
    c      = ccc(y, yh)
    mae    = mean_absolute_error(y, yh)
    rmse   = float(np.sqrt(np.mean((yh - y) ** 2)))
    bias   = float(np.mean(yh - y))
    return dict(label=label, n=len(y), ccc=c, r=r, rho=rho,
                mae=mae, rmse=rmse, bias=bias,
                pred_mean=float(yh.mean()), pred_sd=float(yh.std()),
                gt_mean=float(y.mean()), gt_sd=float(y.std()))

variants = [
    (ROOT / 'results/vc_api_63_v0/vc_scores.csv',           'V0'),
    (ROOT / 'results/vc_api_63_vanilla/vc_scores.csv',       'V0-Vanilla'),
    (ROOT / 'results/vc_api_63_v0_tw_dyn/vc_scores.csv',    'V0+TW-dyn'),
    (ROOT / 'results/vc_api_63_vanilla_tw_dyn/vc_scores.csv','VTW-Vanilla'),
]

rows = []
for p, lbl in variants:
    if Path(p).exists():
        rows.append(metrics(p, lbl))
    else:
        print(f'MISSING: {p}')

HDR = f"{'Variant':<14}  {'n':>3}  {'CCC':>6}  {'r':>6}  {'rho':>6}  {'MAE':>6}  {'RMSE':>6}  {'Bias':>7}  {'predMean':>9}  {'predSD':>7}"
SEP = '-' * len(HDR)
print(HDR)
print(SEP)
for r in rows:
    print(f"{r['label']:<14}  {r['n']:>3}  {r['ccc']:>6.3f}  {r['r']:>6.3f}  {r['rho']:>6.3f}  "
          f"{r['mae']:>6.3f}  {r['rmse']:>6.3f}  {r['bias']:>+7.4f}  "
          f"{r['pred_mean']:>9.3f}  {r['pred_sd']:>7.3f}")

# Pairwise deltas
pairs = [
    ('Role vs no-role (V0)',      'V0',        'V0-Vanilla'),
    ('Role vs no-role (TW-dyn)',  'V0+TW-dyn', 'VTW-Vanilla'),
    ('V0 vs V0+TW-dyn',          'V0',        'V0+TW-dyn'),
    ('Vanilla vs Vanilla+TW',    'V0-Vanilla', 'VTW-Vanilla'),
]
by_label = {r['label']: r for r in rows}
print()
print(f"{'Comparison':<32}  {'ΔCCC':>7}  {'Δr':>7}  {'Δrho':>7}  {'ΔMAE':>7}  {'ΔRMSE':>7}")
print('-' * 72)
for title, a, b in pairs:
    if a not in by_label or b not in by_label:
        continue
    ra, rb = by_label[a], by_label[b]
    print(f"{title:<32}  {ra['ccc']-rb['ccc']:>+7.3f}  {ra['r']-rb['r']:>+7.3f}  "
          f"{ra['rho']-rb['rho']:>+7.3f}  {ra['mae']-rb['mae']:>+7.3f}  {ra['rmse']-rb['rmse']:>+7.3f}")

