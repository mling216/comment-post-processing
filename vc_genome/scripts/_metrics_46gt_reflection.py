"""Pilot metrics for V0+TW vs V0+TW+R vs V0+TW-rsn on the 43-image eval set."""
import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(__file__).parent.parent.parent
ANCHORS = {'VisC.503.6.png', 'InfoVisJ.619.17.png', 'InfoVisJ.1149.6(1).png'}

def ccc(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
    s = np.cov(y, yhat, ddof=0)[0, 1]
    return (2 * s) / (y.var() + yhat.var() + (y.mean() - yhat.mean()) ** 2)

gt = pd.read_csv(ROOT / 'Claude_vc_prediction/gt_all_46.csv')

def metrics(scores_csv, label):
    df = pd.read_csv(scores_csv).rename(columns={'filename': 'imageName'})
    m = gt.merge(df, on='imageName')
    m = m[~m['imageName'].isin(ANCHORS)]
    y = m['NormalizedVC'].values
    yh = m['vc_score'].astype(float).values
    r, p = stats.pearsonr(y, yh)
    rho, _ = stats.spearmanr(y, yh)
    print(f'{label:<14} n={len(y)} CCC={ccc(y,yh):.3f} r={r:.3f} (p={p:.2e}) rho={rho:.3f} '
          f'R2={r2_score(y,yh):.3f} MAE={mean_absolute_error(y,yh):.3f} '
          f'RMSE={np.sqrt(np.mean((yh-y)**2)):.3f} Bias={np.mean(yh-y):+.4f} '
          f'predSD={yh.std():.3f} predMean={yh.mean():.3f} GTmean={y.mean():.3f} GTSD={y.std():.3f}')
    return m

metrics(ROOT / 'results/vc_api_46gt_v0_tw_dyn/vc_scores.csv',     'V0+TW (base)')
metrics(ROOT / 'results/vc_api_46gt_v0_twdyn_r/vc_scores.csv',    'V0+TW+R')
metrics(ROOT / 'results/vc_api_46gt_v0_twdyn_rsn/vc_scores.csv',  'V0+TW-rsn')
