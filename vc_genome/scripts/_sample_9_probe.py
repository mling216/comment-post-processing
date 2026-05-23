import pandas as pd

inp = pd.read_csv('../results/vc_api_510_v0_tw_input.csv')
gt = pd.read_csv('../phrase_reduction_v2/image_compiled_phrases.csv')[['imageName','Topics','NormalizedVC','VisType']]
m = inp[['imageName','imageURL']].merge(gt, on='imageName')

rows = []
for vt, grp in m.groupby('VisType'):
    med = grp['NormalizedVC'].median()
    idx = (grp['NormalizedVC'] - med).abs().idxmin()
    rows.append(grp.loc[idx])

sample = pd.DataFrame(rows).sort_values('VisType')
for _, r in sample.iterrows():
    print(f"{r['imageName']:<32} | {r['VisType']:<18} | VC={r['NormalizedVC']:.2f}")
    print(f"  GT Topics: {r['Topics']}")
    print(f"  URL: {r['imageURL']}")
    print()
