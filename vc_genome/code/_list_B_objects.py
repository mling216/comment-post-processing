import json
from collections import Counter

with open('vc_genome_output_full/three_conditions/oar_B_510.json') as f:
    data = json.load(f)

objs = Counter()
for img, oar in data.items():
    for o in oar.get('objects', []):
        objs[o.get('name', '').strip()] += 1

for name, n in sorted(objs.items(), key=lambda x: -x[1]):
    print(f'{n:4d}  {name}')

print(f'\nTotal unique object names: {len(objs)}')
