"""
Compare all vanilla (no role framing) variants on the 63-image pilot set.

Variants: V0, V0+T, V0+C, V0+A, V0+TW-dyn — all without the expert-role sentence.
"""
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
                pred_mean=float(yh.mean()), pred_sd=float(yh.std()))

variants = [
    (ROOT / 'results/vc_api_63_vanilla/vc_scores.csv',        'V0'),
    (ROOT / 'results/vc_api_63_vanilla_t/vc_scores.csv',      'V0+T'),
    (ROOT / 'results/vc_api_63_vanilla_a/vc_scores.csv',      'V0+A'),
    (ROOT / 'results/vc_api_63_vanilla_tw_dyn/vc_scores.csv', 'V0+TW'),
    (ROOT / 'results/vc_api_63_vanilla_twa/vc_scores.csv',    'V0+TWA'),
]

rows = []
missing = []
for p, lbl in variants:
    if Path(p).exists():
        rows.append(metrics(p, lbl))
    else:
        missing.append((lbl, p))

if missing:
    print('MISSING results (not yet run):')
    for lbl, p in missing:
        print(f'  {lbl}: {p}')
    print()

if not rows:
    print('No results to compare yet.')
    raise SystemExit(0)

HDR = f"{'Variant':<10}  {'n':>3}  {'CCC':>6}  {'r':>6}  {'rho':>6}  {'MAE':>6}  {'RMSE':>6}  {'Bias':>7}  {'predMean':>9}  {'predSD':>7}"
SEP = '-' * len(HDR)
print('=== Vanilla (no role framing) — all variants, n=63 pilot ===')
print(HDR)
print(SEP)
for r in rows:
    print(f"{r['label']:<10}  {r['n']:>3}  {r['ccc']:>6.3f}  {r['r']:>6.3f}  {r['rho']:>6.3f}  "
          f"{r['mae']:>6.3f}  {r['rmse']:>6.3f}  {r['bias']:>+7.4f}  "
          f"{r['pred_mean']:>9.3f}  {r['pred_sd']:>7.3f}")

# Component-effect deltas (all relative to V0-vanilla baseline)
by_label = {r['label']: r for r in rows}
baseline = by_label.get('V0')
if baseline and len(rows) > 1:
    print()
    print(f"{'Variant':<10}  {'ΔCCC':>7}  {'Δr':>7}  {'Δrho':>7}  {'ΔMAE':>7}  {'ΔRMSE':>7}  (vs V0 vanilla)")
    print('-' * 60)
    for r in rows:
        if r['label'] == 'V0':
            continue
        print(f"{r['label']:<10}  {r['ccc']-baseline['ccc']:>+7.3f}  {r['r']-baseline['r']:>+7.3f}  "
              f"{r['rho']-baseline['rho']:>+7.3f}  {r['mae']-baseline['mae']:>+7.3f}  "
              f"{r['rmse']-baseline['rmse']:>+7.3f}")
