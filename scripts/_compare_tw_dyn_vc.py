"""Quick VC prediction metrics comparison: V0+TW vs V0+TW-dyn (on 43 eval images)."""
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
ANCHORS = {'VisC.503.6.png', 'InfoVisJ.619.17.png', 'InfoVisJ.1149.6(1).png'}

gt = pd.read_csv(ROOT / 'Claude_vc_prediction/gt_all_46.csv').rename(
    columns={'imageName': 'filename', 'NormalizedVC': 'gt'})
gt_eval = gt[~gt['filename'].isin(ANCHORS)].copy()

def stats(name, pred_csv):
    pred_csv = ROOT / pred_csv
    if not pred_csv.exists():
        print(f'{name:<22}  (file not found)')
        return
    df = pd.read_csv(pred_csv).rename(columns={'vc_score': 'pred'})
    m = gt_eval.merge(df[['filename', 'pred']], on='filename', how='inner')
    y, yh = m['gt'].values, m['pred'].values
    r, _ = pearsonr(y, yh)
    rho, _ = spearmanr(y, yh)
    mae = mean_absolute_error(y, yh)
    rmse = np.sqrt(mean_squared_error(y, yh))
    bias = (yh - y).mean()
    r2 = r2_score(y, yh)
    my, mh = y.mean(), yh.mean()
    vy, vh = y.var(), yh.var()
    cov = ((y - my) * (yh - mh)).mean()
    ccc = 2 * cov / (vy + vh + (my - mh) ** 2)
    print(f'{name:<22}  n={len(m)}  CCC={ccc:.4f}  Pearson={r:.4f}  Spearman={rho:.4f}  '
          f'R2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  Bias={bias:+.4f}')

print(f'{"Variant":<22}  {"n":>3}  {"CCC":>8}  {"Pearson":>9}  {"Spearman":>10}  '
      f'{"R2":>8}  {"MAE":>8}  {"RMSE":>8}  {"Bias":>8}')
print('-' * 110)

# Existing notebook variants (r1 = t=0, opus-4.6)
stats('V0',               'results/vc_api_46gt_v0/vc_scores.csv')
stats('V0+T',             'results/vc_api_46gt_v0_topic/vc_scores.csv')
stats('V0+TW',            'results/vc_api_46gt_v0_tw/vc_scores.csv')
stats('V0+TWA',            'results/vc_api_46gt_v0_twa/vc_scores.csv')
stats('V0+TWdynA',        'results/vc_api_46gt_v0_twdyna/vc_scores.csv')
stats('V0+TWCA',          'results/vc_api_46gt_v0_tcaw/vc_scores.csv')
stats('V0+TWdynCA',       'results/vc_api_46gt_v0_twdynca/vc_scores.csv')
stats('V0+TW-dyn(no-top3)', 'results/vc_api_46gt_v0_tw_dyn/vc_scores.csv')
stats('V3 (=V1 table)',  'results/vc_api_scores/vc_scores.csv')
stats('V3-dyn (t=1)',    'results/vc_api_46gt_v3_dyn/vc_scores.csv')
stats('V3-dyn (t=0)',    'results/vc_api_46gt_v3_dyn_t0/vc_scores.csv')
stats('V0+TWdyn-det',    'results/vc_api_46gt_v0_twdyn_det/vc_scores.csv')
print()
# Topic-selection variants (sonnet-4.6, t=0)
stats('V0+TW (topicsel)', 'results/vc_api_topicsel_v0_tw/vc_scores.csv')
stats('V0+TW-dyn (sonnet)','results/vc_api_topicsel_v0_tw_dyn/vc_scores.csv')
stats('V0+TW-dyn (opus)', 'results/vc_api_topicsel_v0_tw_dyn_opus/vc_scores.csv')
stats('V0+TW-dyn (no-top3)', 'results/vc_api_46gt_v0_tw_dyn/vc_scores.csv')
