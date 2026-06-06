"""
Build oar_B_synset_object_mapping.csv
A flat table showing how each raw object name maps to its synset term and category.

Output columns:
  raw_name        - the original free-form name extracted by OAR-B
  synset_term     - the subcategory (leaf) in the synset  (e.g. "legend")
  synset_category - the category (group) in the synset    (e.g. "scaffold")
  n_images        - how many images this raw_name appears in

Output: vc_genome/export/oar_B_synset_object_mapping.csv
"""

import re
import pandas as pd
from collections import Counter

TRACEABILITY = "vc_genome_output_full/vistype_profile/oar_image_traceability.csv"

# Pattern: "raw_name --> category.subcategory [region]"
OBJ_PATTERN = re.compile(r"(.+?)\s*-->\s*(\w+)\.(\w+)\s*\[.+?\]")

df = pd.read_csv(TRACEABILITY)

counts = Counter()   # (raw_name, synset_term, synset_category) -> n_images

for _, row in df.iterrows():
    obj_str = row.get('objects', '')
    if not isinstance(obj_str, str) or not obj_str.strip():
        continue
    for part in obj_str.split(' | '):
        m = OBJ_PATTERN.match(part.strip())
        if m:
            raw_name = m.group(1).strip()
            category = m.group(2).strip()
            subcategory = m.group(3).strip()
            counts[(raw_name, subcategory, category)] += 1

# Rename legacy category label from OAR-B extraction
CATEGORY_RENAME = {'furniture': 'scaffold'}

rows = [
    {'raw_name': raw, 'synset_term': term, 'synset_category': CATEGORY_RENAME.get(cat, cat), 'n_images': n}
    for (raw, term, cat), n in counts.items()
]

result = (pd.DataFrame(rows)
          .sort_values(['synset_category', 'synset_term', 'n_images'], ascending=[True, True, False])
          .reset_index(drop=True))

output = "vc_genome/export/oar_B_synset_object_mapping.csv"
result.to_csv(output, index=False)
print(f"Saved {len(result)} rows to {output}")
print(f"  Unique raw names   : {result['raw_name'].nunique()}")
print(f"  Unique synset terms: {result['synset_term'].nunique()}")
print(f"\nSample:")
print(result.groupby(['synset_category', 'synset_term'])
      .apply(lambda g: g.nlargest(2, 'n_images'))
      .reset_index(drop=True)
      .to_string(index=False))
