import json, pandas as pd
from collections import defaultdict

b_raw = json.load(open('vc_genome_output_full/three_conditions/oar_B_510.json', encoding='utf-8'))

name_to_images = defaultdict(list)
for img, ext in b_raw.items():
    for obj in ext.get('objects', []):
        name_to_images[obj['name']].append(img)

print(f'Unique raw object names in oar_B_510.json: {len(name_to_images)}')
ex = [(n, imgs) for n, imgs in name_to_images.items() if len(imgs) > 5][:3]
for n, imgs in ex:
    print(f'  "{n}" appears in {len(imgs)} images: {imgs[:3]}...')

# Cross-check with synset_mappings_B
sm = pd.read_csv('vc_genome_output_full/vistype_profile/synset_mappings_B.csv')
in_json = set(name_to_images.keys())
in_csv  = set(sm['raw_name'])
print(f'\nraw_names in synset_mappings_B: {len(in_csv)}')
print(f'raw_names also in oar_B_510.json: {len(in_csv & in_json)}')
print(f'raw_names in synset_mappings_B but NOT in oar_B_510.json: {len(in_csv - in_json)}')
