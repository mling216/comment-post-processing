import json
from collections import Counter

with open('vc_genome_output_full/three_conditions/oar_B_510.json') as f:
    data = json.load(f)

topics = Counter()
for img, oar in data.items():
    for attr in oar.get('attributes', []):
        t = attr.get('topic', '').strip()
        topics[t] += 1

for t, n in sorted(topics.items(), key=lambda x: -x[1]):
    print(f'{n:4d}  {repr(t)}')
