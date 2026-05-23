"""
Compare V0+T on 63-image pilot vs the 43-image original result.

The 43-image result (from tab:full in the paper) is hard-coded below.
The 63-image result is read from results/vc_api_63_v0_t/vc_scores.csv.
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

def metrics_from_file(path, label, exclude_anchors=True):
    df = pd.read_csv(path).rename(columns={'filename': 'imageName'})
    m  = gt.merge(df, on='imageName')
    if exclude_anchors:
        m = m[~m['imageName'].isin(ANCHORS)]
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

# ── Published 43-image result from tab:full (paper appendix) ─────────────────
PAPER_43 = dict(label='V0+T (43-img, published)', n=43,
                ccc=0.869, r=0.879, rho=0.872, mae=0.078, rmse=0.095, bias=+0.003,
                pred_mean=float('nan'), pred_sd=float('nan'))

# ── Live 63-image run ─────────────────────────────────────────────────────────
scores_path = ROOT / 'results/vc_api_63_v0_t/vc_scores.csv'

rows = [PAPER_43]
if scores_path.exists():
    rows.append(metrics_from_file(scores_path, 'V0+T (63-img, new)'))
else:
    print(f'MISSING: {scores_path}')
    print('Run first:')
    print('  python _vc_score_api_v0_topic.py --input-csv ../Claude_vc_prediction/gt_all_66.csv '
          '--outdir ../results/vc_api_63_v0_t --concurrency 5 --model claude-opus-4-6')
    raise SystemExit(1)

HDR = f"{'Variant':<30}  {'n':>3}  {'CCC':>6}  {'r':>6}  {'rho':>6}  {'MAE':>6}  {'RMSE':>6}  {'Bias':>7}"
SEP = '-' * len(HDR)
print(HDR)
print(SEP)
for r in rows:
    print(f"{r['label']:<30}  {r['n']:>3}  {r['ccc']:>6.3f}  {r['r']:>6.3f}  {r['rho']:>6.3f}  "
          f"{r['mae']:>6.3f}  {r['rmse']:>6.3f}  {r['bias']:>+7.4f}")

if len(rows) == 2:
    a, b = rows[0], rows[1]
    print()
    print(f"{'Delta (63 - 43)':<30}  {'':>3}  {b['ccc']-a['ccc']:>+6.3f}  {b['r']-a['r']:>+6.3f}  "
          f"{b['rho']-a['rho']:>+6.3f}  {b['mae']-a['mae']:>+6.3f}  {b['rmse']-a['rmse']:>+6.3f}  "
          f"{b['bias']-a['bias']:>+7.4f}")
