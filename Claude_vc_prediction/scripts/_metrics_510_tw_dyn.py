"""Compute full-set metrics for V0+TW-dyn vs V0+TW and V1 on the 510-image set."""
import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(__file__).parent.parent.parent

def ccc(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
    s_yyhat = np.cov(y, yhat, ddof=0)[0, 1]
    return (2 * s_yyhat) / (y.var() + yhat.var() + (y.mean() - yhat.mean())**2)

gt_df = pd.read_csv(ROOT / 'phrase_reduction_v2/image_compiled_phrases.csv')[['imageName', 'NormalizedVC']].drop_duplicates('imageName')
ANCHORS = {'VisC.503.6.png', 'InfoVisJ.619.17.png', 'InfoVisJ.1149.6(1).png'}

def run_metrics(scores_csv, label):
    df = pd.read_csv(scores_csv).rename(columns={'filename': 'imageName'})
    merged = gt_df.merge(df, on='imageName')
    merged = merged[~merged['imageName'].isin(ANCHORS)]
    y = merged['NormalizedVC'].values
    yhat = merged['vc_score'].values
    r_val, _ = stats.pearsonr(y, yhat)
    rho_val, _ = stats.spearmanr(y, yhat)
    r2 = r2_score(y, yhat)
    mae = mean_absolute_error(y, yhat)
    rmse = np.sqrt(np.mean((yhat - y)**2))
    bias = np.mean(yhat - y)
    c = ccc(y, yhat)
    print(f"{label}: n={len(y)}, CCC={c:.3f}, r={r_val:.3f}, rho={rho_val:.3f}, R2={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}, Bias={bias:+.3f}")
    print(f"  pred_mean={yhat.mean():.3f}, pred_SD={yhat.std():.3f}, GT_mean={y.mean():.3f}, GT_SD={y.std():.3f}")
    return merged

run_metrics(ROOT / 'results/vc_api_510_v0_tw/vc_scores.csv',           'V0+TW         (510)')
m_dyn = run_metrics(ROOT / 'results/vc_api_510_v0_tw_dyn/vc_scores.csv',   'V0+TW-dyn     (510)')
run_metrics(ROOT / 'results/vc_api_510_v3_t0/vc_scores.csv',           'V1 t=0        (507)')
run_metrics(ROOT / 'results/vc_api_510_v3_dyn_t0/vc_scores.csv',       'V1-dyn t=0    (507)')
run_metrics(ROOT / 'results/vc_api_510_v0_twdyn_det_t0/vc_scores.csv', 'V0+TWdyn-det  (510)')

# Per-vis-type for V0+TW-dyn vs V0+TW
print("\n--- Per-vis-type: V0+TW-dyn vs V0+TW ---")
gt_full = pd.read_csv(ROOT / 'phrase_reduction_v2/image_compiled_phrases.csv')[['imageName','NormalizedVC','VisType']].drop_duplicates('imageName')

tw_df     = pd.read_csv(ROOT / 'results/vc_api_510_v0_tw/vc_scores.csv').rename(columns={'filename':'imageName','vc_score':'vc_tw'})
twdyn_df  = pd.read_csv(ROOT / 'results/vc_api_510_v0_tw_dyn/vc_scores.csv').rename(columns={'filename':'imageName','vc_score':'vc_twdyn'})
v1_df     = pd.read_csv(ROOT / 'results/vc_api_510_v3_t0/vc_scores.csv').rename(columns={'filename':'imageName','vc_score':'vc_v1'})
v1dyn_df  = pd.read_csv(ROOT / 'results/vc_api_510_v3_dyn_t0/vc_scores.csv').rename(columns={'filename':'imageName','vc_score':'vc_v1dyn'})
twddet_df = pd.read_csv(ROOT / 'results/vc_api_510_v0_twdyn_det_t0/vc_scores.csv').rename(columns={'filename':'imageName','vc_score':'vc_twddet'})

merged = (gt_full.merge(tw_df, on='imageName')
                 .merge(twdyn_df, on='imageName')
                 .merge(v1_df, on='imageName')
                 .merge(v1dyn_df, on='imageName')
                 .merge(twddet_df, on='imageName'))
merged = merged[~merged['imageName'].isin(ANCHORS)]

print(f"\n{'VisType':<22} | {'n':>3} | {'CCC_TW':>6} | {'CCC_dyn':>7} | {'DCCC':>6} | {'R2_TW':>6} | {'R2_dyn':>6} | {'MAE_TW':>6} | {'MAE_dyn':>7}")
for vt, grp in merged.groupby('VisType'):
    y = grp['NormalizedVC'].values
    ytw   = grp['vc_tw'].values
    ydyn  = grp['vc_twdyn'].values
    c_tw   = ccc(y, ytw);  c_dyn  = ccc(y, ydyn)
    r2_tw  = r2_score(y, ytw); r2_dyn = r2_score(y, ydyn)
    mae_tw = mean_absolute_error(y, ytw); mae_dyn = mean_absolute_error(y, ydyn)
    print(f"{vt:<22} | {len(y):>3} | {c_tw:>6.3f} | {c_dyn:>7.3f} | {c_dyn-c_tw:>+6.3f} | {r2_tw:>6.3f} | {r2_dyn:>6.3f} | {mae_tw:>6.3f} | {mae_dyn:>7.3f}")

print(f"\n--- Per-vis-type: V1-dyn vs V1 (t=0) ---")
print(f"{'VisType':<22} | {'n':>3} | {'CCC_V1':>6} | {'CCC_dyn':>7} | {'DCCC':>6} | {'R2_V1':>6} | {'R2_dyn':>6} | {'MAE_V1':>6} | {'MAE_dyn':>7}")
for vt, grp in merged.groupby('VisType'):
    y = grp['NormalizedVC'].values
    yv1   = grp['vc_v1'].values
    ydyn  = grp['vc_v1dyn'].values
    c_v1   = ccc(y, yv1);  c_dyn  = ccc(y, ydyn)
    r2_v1  = r2_score(y, yv1); r2_dyn = r2_score(y, ydyn)
    mae_v1 = mean_absolute_error(y, yv1); mae_dyn = mean_absolute_error(y, ydyn)
    print(f"{vt:<22} | {len(y):>3} | {c_v1:>6.3f} | {c_dyn:>7.3f} | {c_dyn-c_v1:>+6.3f} | {r2_v1:>6.3f} | {r2_dyn:>6.3f} | {mae_v1:>6.3f} | {mae_dyn:>7.3f}")

print(f"\n--- Per-vis-type: V0+TWdyn-det vs V1 (t=0) ---")
print(f"{'VisType':<22} | {'n':>3} | {'CCC_V1':>6} | {'CCC_det':>7} | {'DCCC':>6} | {'R2_V1':>6} | {'R2_det':>6} | {'MAE_V1':>6} | {'MAE_det':>7}")
for vt, grp in merged.groupby('VisType'):
    y = grp['NormalizedVC'].values
    yv1   = grp['vc_v1'].values
    ydet  = grp['vc_twddet'].values
    c_v1   = ccc(y, yv1);  c_det  = ccc(y, ydet)
    r2_v1  = r2_score(y, yv1); r2_det = r2_score(y, ydet)
    mae_v1 = mean_absolute_error(y, yv1); mae_det = mean_absolute_error(y, ydet)
    print(f"{vt:<22} | {len(y):>3} | {c_v1:>6.3f} | {c_det:>7.3f} | {c_det-c_v1:>+6.3f} | {r2_v1:>6.3f} | {r2_det:>6.3f} | {mae_v1:>6.3f} | {mae_det:>7.3f}")

print(f"\n--- Per-vis-type: V0+TWdyn-det vs V0+TW-dyn (isolates per-dim output) ---")
print(f"{'VisType':<22} | {'n':>3} | {'CCC_dyn':>7} | {'CCC_det':>7} | {'DCCC':>6} | {'R2_dyn':>6} | {'R2_det':>6} | {'MAE_dyn':>7} | {'MAE_det':>7}")
for vt, grp in merged.groupby('VisType'):
    y = grp['NormalizedVC'].values
    ydyn  = grp['vc_twdyn'].values
    ydet  = grp['vc_twddet'].values
    c_dyn  = ccc(y, ydyn); c_det  = ccc(y, ydet)
    r2_dyn = r2_score(y, ydyn); r2_det = r2_score(y, ydet)
    mae_dyn = mean_absolute_error(y, ydyn); mae_det = mean_absolute_error(y, ydet)
    print(f"{vt:<22} | {len(y):>3} | {c_dyn:>7.3f} | {c_det:>7.3f} | {c_det-c_dyn:>+6.3f} | {r2_dyn:>6.3f} | {r2_det:>6.3f} | {mae_dyn:>7.3f} | {mae_det:>7.3f}")
