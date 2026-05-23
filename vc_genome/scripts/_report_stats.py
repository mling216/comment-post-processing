"""Gather numbers for the OAR analysis report."""
import json, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path('vc_genome_output_full')
per = pd.read_csv(ROOT / 'match/per_image_metrics.csv')

b  = json.loads((ROOT / 'three_conditions/oar_B.json').read_text())
v1 = json.loads((ROOT / 'three_conditions/oar_V1.json').read_text())
v2 = json.loads((ROOT / 'three_conditions/oar_V2.json').read_text())

common   = sorted(set(b) & set(v1))
common_v2 = sorted(set(b) & set(v2))

print("=== B vs V1 distribution (n=63) ===")
for label, data, imgs in [('B', b, common), ('V1', v1, common), ('V2', v2, common_v2)]:
    o  = np.mean([len(data[i]['objects'])      for i in imgs])
    a  = np.mean([len(data[i]['attributes'])   for i in imgs])
    r  = np.mean([len(data[i]['relationships']) for i in imgs])
    print(f"  {label}: obj={o:.1f}  attr={a:.1f}  rel={r:.1f}")

print()
print("=== Agreement metrics (B vs V1, n=63) ===")
cols = ['obj_jaccard','obj_precision','obj_recall','obj_f1',
        'attr_jaccard','attr_f1',
        'rel_strict_jaccard','rel_strict_f1',
        'rel_loose_jaccard','rel_loose_f1']
for c in cols:
    print(f"  {c:28s} mean={per[c].mean():.3f}  median={per[c].median():.3f}  sd={per[c].std():.3f}")

print()
print("=== Top-3 highest agreement images (by obj_f1) ===")
print(per.nlargest(3,'obj_f1')[['imageName','obj_f1','attr_f1','rel_strict_f1']].to_string(index=False))
print()
print("=== Top-3 lowest agreement images (by obj_f1) ===")
print(per[per['obj_f1'] > 0].nsmallest(3,'obj_f1')[['imageName','obj_f1','attr_f1','rel_strict_f1']].to_string(index=False))
