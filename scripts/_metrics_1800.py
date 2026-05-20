import pandas as pd, numpy as np
from scipy import stats
from sklearn.metrics import r2_score

BASE = r'd:\Coding\Copilot\comment_post_processing'
gt = pd.read_csv(f'{BASE}/Claude_vc_prediction/FormalExp1800Images.csv')[['ImageName','NormalizedVC','VisTypes']]

def ccc(y, yh):
    co = np.cov(y, yh)[0, 1]
    return 2 * co / (y.var() + yh.var() + (y.mean() - yh.mean()) ** 2)

runs = [
    ('V0',         f'{BASE}/results/vc_api_1800_v0/vc_scores.csv'),
    ('V0+TW_dyn',  f'{BASE}/results/vc_api_1800_v0_tw_dyn/vc_scores.csv'),
]

print(f"{'Variant':<12}  {'n':>4}  {'CCC':>6}  {'r':>6}  {'rho':>6}  {'R2':>6}  {'MAE':>6}  {'RMSE':>6}  {'Bias':>7}")
print('-' * 72)
for name, path in runs:
    pred = pd.read_csv(path).rename(columns={'filename': 'ImageName', 'vc_score': 'pred'})
    df = gt.merge(pred, on='ImageName')
    y, yh = df['NormalizedVC'].values, df['pred'].values
    r, _   = stats.pearsonr(y, yh)
    rho, _ = stats.spearmanr(y, yh)
    mae    = np.mean(np.abs(yh - y))
    rmse   = np.sqrt(np.mean((yh - y) ** 2))
    bias   = np.mean(yh - y)
    r2     = r2_score(y, yh)
    print(f"{name:<12}  {len(df):>4}  {ccc(y,yh):>6.3f}  {r:>6.3f}  {rho:>6.3f}  {r2:>6.3f}  {mae:>6.3f}  {rmse:>6.3f}  {bias:>+7.4f}")
