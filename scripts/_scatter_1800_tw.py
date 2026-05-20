"""Scatter plot: predicted vs. ground-truth VC for the 1800-image VIS2025 set."""
import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent

def ccc(y, yhat):
    s = np.cov(y, yhat, ddof=0)[0, 1]
    return (2 * s) / (y.var() + yhat.var() + (y.mean() - yhat.mean()) ** 2)

gt = pd.read_csv(ROOT / 'Claude_vc_prediction/FormalExp1800Images.csv')[
    ['ImageName', 'NormalizedVC', 'VisTypes', 'Dataset']
].rename(columns={'ImageName': 'imageName'}).drop_duplicates('imageName')
scores = pd.read_csv(ROOT / 'results/vc_api_1800_v0_tw_dyn/vc_scores.csv').rename(
    columns={'filename': 'imageName', 'vc_score': 'pred'})
scores['pred'] = scores['pred'].astype(float)
m = gt.merge(scores, on='imageName')
# normalize casing glitch: lower 'schematic' -> 'Schematic'
m['VisTypes'] = m['VisTypes'].str.strip().replace({'schematic': 'Schematic'})

y, yh = m.NormalizedVC.values, m.pred.values
r, _ = stats.pearsonr(y, yh)
rho, _ = stats.spearmanr(y, yh)
c, mae = ccc(y, yh), np.mean(np.abs(yh - y))
r2 = r2_score(y, yh)

# Color-by-vistype scatter (top-9 types) + identity & regression line
top_types = m['VisTypes'].value_counts().head(9).index.tolist()
cmap = plt.get_cmap('tab10')
color_map = {vt: cmap(i) for i, vt in enumerate(top_types)}

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

# (a) Overall, colored by vis-type
ax = axes[0]
for vt in top_types:
    sub = m[m['VisTypes'] == vt]
    ax.scatter(sub['NormalizedVC'], sub['pred'], s=14, alpha=0.55,
               color=color_map[vt], label=f'{vt} (n={len(sub)})', edgecolor='none')
other = m[~m['VisTypes'].isin(top_types)]
if len(other):
    ax.scatter(other['NormalizedVC'], other['pred'], s=14, alpha=0.4,
               color='lightgray', label=f'other (n={len(other)})', edgecolor='none')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7, label='identity')
slope, intercept = np.polyfit(y, yh, 1)
xs = np.linspace(0, 1, 100)
ax.plot(xs, slope * xs + intercept, color='red', lw=1.4, alpha=0.8,
        label=f'fit: $\\hat{{y}}={slope:.2f}y{intercept:+.2f}$')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel('Ground-truth NormalizedVC'); ax.set_ylabel('Predicted vc_score (V0+TW)')
ax.set_title(f'V0+TW on 1800-image VIS2025 set\n'
             f'CCC={c:.3f}  r={r:.3f}  $\\rho$={rho:.3f}  $R^2$={r2:.3f}  MAE={mae:.3f}  n={len(m)}')
ax.legend(loc='upper left', fontsize=8, ncol=2, frameon=True)
ax.grid(alpha=0.25)

# (b) Hexbin density + identity
ax = axes[1]
hb = ax.hexbin(y, yh, gridsize=35, cmap='viridis', mincnt=1)
ax.plot([0, 1], [0, 1], 'w--', lw=1.2, alpha=0.9)
ax.plot(xs, slope * xs + intercept, color='orange', lw=1.4, alpha=0.95)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel('Ground-truth NormalizedVC'); ax.set_ylabel('Predicted vc_score')
ax.set_title('Density (hexbin)')
fig.colorbar(hb, ax=ax, label='count')
ax.grid(alpha=0.2, color='white')

plt.tight_layout()
out_png = ROOT / 'results/vc_api_1800_v0_tw_dyn/scatter_1800_v0tw.png'
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f'Saved: {out_png}')

# Per-vis-type small-multiples
fig2, axes2 = plt.subplots(3, 3, figsize=(11, 10), sharex=True, sharey=True)
for ax, vt in zip(axes2.flat, top_types):
    sub = m[m['VisTypes'] == vt]
    yy, yhh = sub['NormalizedVC'].values, sub['pred'].values
    rr, _ = stats.pearsonr(yy, yhh)
    cc = ccc(yy, yhh)
    rr2 = r2_score(yy, yhh)
    ax.scatter(yy, yhh, s=10, alpha=0.55, color=color_map[vt], edgecolor='none')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.9, alpha=0.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f'{vt}  (n={len(sub)})\nCCC={cc:.2f}  r={rr:.2f}  $R^2$={rr2:.2f}', fontsize=10)
    ax.grid(alpha=0.2)
for ax in axes2[-1, :]: ax.set_xlabel('GT VC')
for ax in axes2[:, 0]:  ax.set_ylabel('Pred VC')
fig2.suptitle('V0+TW on 1800-image VIS2025: per-vis-type', fontsize=12, y=1.00)
plt.tight_layout()
out_png2 = ROOT / 'results/vc_api_1800_v0_tw_dyn/scatter_1800_v0tw_per_vistype.png'
fig2.savefig(out_png2, dpi=150, bbox_inches='tight')
print(f'Saved: {out_png2}')
