"""Full-set (n=507) metrics: V0+TW vs V0+TW+R."""
import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(__file__).parent.parent
ANCHORS = {'VisC.503.6.png', 'InfoVisJ.619.17.png', 'InfoVisJ.1149.6(1).png'}

def ccc(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
    s = np.cov(y, yhat, ddof=0)[0, 1]
    return (2 * s) / (y.var() + yhat.var() + (y.mean() - yhat.mean()) ** 2)

gt_full = pd.read_csv(ROOT / 'phrase_reduction_v2/image_compiled_phrases.csv')[
    ['imageName', 'NormalizedVC', 'VisType']
].drop_duplicates('imageName')

def metrics(scores_csv, label):
    df = pd.read_csv(scores_csv).rename(columns={'filename': 'imageName'})
    m = gt_full.merge(df, on='imageName')
    m = m[~m['imageName'].isin(ANCHORS)]
    y = m['NormalizedVC'].values
    yh = m['vc_score'].astype(float).values
    r, p = stats.pearsonr(y, yh)
    rho, _ = stats.spearmanr(y, yh)
    print(f'{label:<14} n={len(y)} CCC={ccc(y,yh):.3f} r={r:.3f} (p={p:.2e}) rho={rho:.3f} '
          f'R2={r2_score(y,yh):.3f} MAE={mean_absolute_error(y,yh):.3f} '
          f'RMSE={np.sqrt(np.mean((yh-y)**2)):.3f} Bias={np.mean(yh-y):+.4f} '
          f'predSD={yh.std():.3f} predMean={yh.mean():.3f}')
    return m

m_tw = metrics(ROOT / 'results/vc_api_510_v0_tw_dyn/vc_scores.csv',  'V0+TW (510)')
m_r  = metrics(ROOT / 'results/vc_api_510_v0_twdyn_r/vc_scores.csv', 'V0+TW+R (510)')

# Per-vis-type
tw  = pd.read_csv(ROOT / 'results/vc_api_510_v0_tw_dyn/vc_scores.csv').rename(columns={'filename':'imageName','vc_score':'vc_tw'})
r_  = pd.read_csv(ROOT / 'results/vc_api_510_v0_twdyn_r/vc_scores.csv').rename(columns={'filename':'imageName','vc_score':'vc_r'})

merged = gt_full.merge(tw, on='imageName').merge(r_, on='imageName')
merged = merged[~merged['imageName'].isin(ANCHORS)]
merged['vc_tw'] = merged['vc_tw'].astype(float)
merged['vc_r']  = merged['vc_r'].astype(float)

print(f"\n--- Per-vis-type: V0+TW+R vs V0+TW ---")
print(f"{'VisType':<22} | {'n':>3} | {'CCC_TW':>6} | {'CCC_R':>6} | {'DCCC':>6} | {'R2_TW':>6} | {'R2_R':>6} | {'MAE_TW':>6} | {'MAE_R':>6}")
rows = []
for vt, grp in merged.groupby('VisType'):
    y = grp['NormalizedVC'].values
    ytw = grp['vc_tw'].values; yr = grp['vc_r'].values
    c_tw = ccc(y, ytw); c_r = ccc(y, yr)
    r2_tw = r2_score(y, ytw); r2_r = r2_score(y, yr)
    mae_tw = mean_absolute_error(y, ytw); mae_r = mean_absolute_error(y, yr)
    rows.append((vt, len(y), c_tw, c_r, c_r-c_tw, r2_tw, r2_r, mae_tw, mae_r))
    print(f"{vt:<22} | {len(y):>3} | {c_tw:>6.3f} | {c_r:>6.3f} | {c_r-c_tw:>+6.3f} | {r2_tw:>6.3f} | {r2_r:>6.3f} | {mae_tw:>6.3f} | {mae_r:>6.3f}")
