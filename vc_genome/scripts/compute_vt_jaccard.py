import pandas as pd

def jaccard(a, b):
    sa = set(str(a).split(';')) if pd.notna(a) and str(a).strip() else set()
    sb = set(str(b).split(';')) if pd.notna(b) and str(b).strip() else set()
    if not sa and not sb: return 1.0
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

gt = pd.read_csv('results/probe_prominent_63/analysis_63.csv')[['filename','gt','vistype']]

for folder, label in [('vc_api_topicsel_v0_t_vt','V0+T+VT'), ('vc_api_topicsel_v0_tw_dyn_opus','V0+TW+VT')]:
    preds = pd.read_csv(f'results/{folder}/vc_scores.csv')[['filename','top3_topics']]
    merged = gt.merge(preds, on='filename')
    merged['j'] = merged.apply(lambda r: jaccard(r['top3_topics'], r['gt']), axis=1)
    macro = round(merged['j'].mean(), 3)
    per_vt = merged.groupby('vistype')['j'].mean().round(3)
    print(f'{label}: macro J={macro}, n={len(merged)}')
    for v, j in sorted(per_vt.items()):
        print(f'  {v}: {j}')
