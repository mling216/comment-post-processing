"""V0+TW on the 1800-image VIS2025 dataset."""
import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(__file__).parent.parent.parent

def ccc(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
    s = np.cov(y, yhat, ddof=0)[0, 1]
    return (2 * s) / (y.var() + yhat.var() + (y.mean() - yhat.mean()) ** 2)

gt = pd.read_csv(ROOT / 'Claude_vc_prediction/FormalExp1800Images.csv')[
    ['ImageName', 'NormalizedVC', 'VisTypes', 'Dataset']
].rename(columns={'ImageName': 'imageName'}).drop_duplicates('imageName')

scores = pd.read_csv(ROOT / 'results/vc_api_1800_v0_tw_dyn/vc_scores.csv').rename(
    columns={'filename': 'imageName', 'vc_score': 'pred'})
scores['pred'] = scores['pred'].astype(float)

m = gt.merge(scores, on='imageName')
print(f'Merged: {len(m)} / {len(gt)} GT rows  ({len(scores)} predictions)')

# GT stats
print(f'\nGT: mean={m.NormalizedVC.mean():.3f} SD={m.NormalizedVC.std():.3f} '
      f'min={m.NormalizedVC.min():.3f} max={m.NormalizedVC.max():.3f}')
print(f'Pred: mean={m.pred.mean():.3f} SD={m.pred.std():.3f} '
      f'min={m.pred.min():.3f} max={m.pred.max():.3f}')

y, yh = m.NormalizedVC.values, m.pred.values
r, p = stats.pearsonr(y, yh)
rho, _ = stats.spearmanr(y, yh)
print(f'\nOverall (n={len(y)}):')
print(f'  CCC   = {ccc(y, yh):.4f}')
print(f'  r     = {r:.4f}  (p = {p:.2e})')
print(f'  rho   = {rho:.4f}')
print(f'  R2    = {r2_score(y, yh):.4f}')
print(f'  MAE   = {mean_absolute_error(y, yh):.4f}')
print(f'  RMSE  = {np.sqrt(np.mean((yh - y) ** 2)):.4f}')
print(f'  Bias  = {np.mean(yh - y):+.4f}')

# Per-vis-type
print(f'\n--- Per VisType ---')
print(f"{'VisType':<22} | {'n':>4} | {'CCC':>6} | {'r':>6} | {'rho':>6} | {'R2':>7} | {'MAE':>6} | {'Bias':>7}")
for vt, grp in m.groupby('VisTypes'):
    if len(grp) < 5:
        continue
    yy, yhh = grp.NormalizedVC.values, grp.pred.values
    rr, _ = stats.pearsonr(yy, yhh)
    rrho, _ = stats.spearmanr(yy, yhh)
    print(f'{vt:<22} | {len(grp):>4} | {ccc(yy, yhh):>6.3f} | {rr:>6.3f} | {rrho:>6.3f} | '
          f'{r2_score(yy, yhh):>7.3f} | {mean_absolute_error(yy, yhh):>6.3f} | {np.mean(yhh-yy):>+7.4f}')

# Per-dataset
print(f'\n--- Per Dataset ---')
print(f"{'Dataset':<15} | {'n':>4} | {'CCC':>6} | {'r':>6} | {'R2':>7} | {'MAE':>6} | {'Bias':>7}")
for ds, grp in m.groupby('Dataset'):
    yy, yhh = grp.NormalizedVC.values, grp.pred.values
    rr, _ = stats.pearsonr(yy, yhh)
    print(f'{ds:<15} | {len(grp):>4} | {ccc(yy, yhh):>6.3f} | {rr:>6.3f} | {r2_score(yy, yhh):>7.3f} | '
          f'{mean_absolute_error(yy, yhh):>6.3f} | {np.mean(yhh-yy):>+7.4f}')
