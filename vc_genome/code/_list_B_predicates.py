import json
from collections import Counter

with open('vc_genome_output_full/three_conditions/oar_B_510.json') as f:
    data = json.load(f)

preds = Counter()
for img, oar in data.items():
    for rel in oar.get('relationships', []):
        if 'pred' in rel:
            preds[rel['pred']] += 1

for p, n in sorted(preds.items()):
    print(f'{n:3d}  {p}')

print(f'\nTotal unique predicates: {len(preds)}')
