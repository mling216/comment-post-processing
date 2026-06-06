"""
Build a three-level dictionary from OAR B synset and raw JSON data.

Sections:
  visual_objects     -> subcategory (synset category) -> term
  visual_attributes  -> subcategory (synset category) -> term
  visual_predicates  -> subcategory (semantic class)  -> predicate term

The predicate subcategories mirror the pure_genome taxonomy:
  spatial     - positional/locational relationships
  semantic    - encoding, representation, annotation
  structural  - composition, combination, organization
  perceptual  - visual appearance, visibility effects
  cognitive   - interpretation, understanding, comprehension
  evaluative  - complexity/clutter contribution, requirements

Outputs: vc_genome/export/oar_B_dict_flat.csv
"""

import re
import json
import pandas as pd
from collections import Counter

# Priority-ordered predicate classification rules.
# Each entry: (subcategory, list-of-substrings-any-of-which-triggers-match)
PRED_RULES = [
    # SPATIAL - where things are relative to each other
    ('spatial', [
        'overlap', 'intersect', 'scatter', 'distribut', 'plotted',
        'arranged_along', 'arranged_in', 'aligned_with',
        'converge', 'cluster', 'bounded_by', 'extends_along', 'packed',
        'span', 'transition', 'embedded', 'fills', 'paired_with',
        'connected_by', 'branches_into', 'too_close', 'part_of',
        'underlies', 'repeated_along', 'shares_time_axis',
        'displayed_on', 'positioned',
    ]),
    # SEMANTIC - encoding, representing, annotating meaning
    ('semantic', [
        'encod', 'represent', 'annotate', 'annotated',
        'communicat', 'convey', 'mapped_to', 'supplement',
        'summariz', 'explains', 'labeled_by', 'labeled_with',
        'labels', 'does_not_encode', 'ambiguously_encodes',
        'communicates_frequency', 'communicates_quickly',
    ]),
    # STRUCTURAL - composition, combination, organization
    ('structural', [
        'contain', 'compos', 'combin', 'differentiat', 'organiz',
        'partitions', 'reinforc', 'rendered_as', 'rendered_in',
        'rendered_with', 'grouped_by', 'mixed_with', 'co_occurs',
        'interact', 'contrasts_with', 'absent_from', 'absence_of',
        'accompanied', 'accompanies', 'associated_with', 'differs_from',
        'outnumber', 'forms', 'supported_by', 'contextualiz',
        'provides_familiar', 'provides_structure', 'provides_information',
        'breaks_down', 'resembles',
    ]),
    # PERCEPTUAL - visual appearance, visibility, what the eye sees
    ('perceptual', [
        'visual_clutter', 'visual_complex', 'visual_disorder',
        'visual_load', 'visual_strain', 'visually_',
        'clutters', 'crowds', 'dominat', 'distracts',
        'draws_attention', 'obscur', 'overwhelm',
        'makes_appear', 'makes_cluttered', 'makes_confusing',
        'makes_easier_to_visualize', 'makes_harder_to_distinguish',
        'makes_harder_to_read', 'makes_harder_to_extract',
        'impedes_distinction', 'overlap_making', 'overlaps_obscuring',
        'perceived_as', 'produces_illusion', 'too_small',
        'more_complex_than', 'simpler_than', 'is_messier_than',
        'similarity_hinders_distinction',
    ]),
    # COGNITIVE - interpretation, understanding, comprehension
    ('cognitive', [
        'clarif', 'comprehend', 'interpret', 'makes_sense',
        'facilitat', 'enables_at_a_glance', 'enables_quick',
        'fails_to_clarif', 'fails_to_convey',
        'hinders_comprehension', 'hinders_interpretation', 'hinders_visibility',
        'aids_interpretation', 'aids_differentiation', 'aids_easy',
        'supports_interpretation', 'supports_readability',
        'supports_quick', 'supports_easy', 'supports_evaluation',
        'affords_quick', 'easy_to_follow',
        'causes_comprehension', 'causes_confusion', 'causes_interpretive',
        'confounds', 'impedes_interpretation',
        'prolongs_decoding', 'prolongs_interpretation',
        'reduces_interpretab', 'reduces_interpretive',
        'simplifies_reading', 'provides_context', 'positively_received',
    ]),
    # EVALUATIVE - effect on clutter, complexity, cognitive load, requirements
    ('evaluative', [
        'increases_', 'reduces_clutter', 'reduces_cognitive',
        'reduces_complexity', 'reduces_difficulty', 'reduces_ambiguity',
        'reduces_contrast', 'reduces_differenti',
        'reduces_distinguishab', 'reduces_legib',
        'reduces_perceived', 'reduces_clarity', 'reduces_visual_clutter',
        'contributes_to', 'contributes_clutter',
        'adds_cognitive', 'adds_complexity', 'adds_density',
        'adds_reading', 'adds_analytical', 'adds_detail',
        'adds_information', 'adds_interpretive', 'adds_nuance',
        'adds_structural', 'adds_visual', 'adds_clutter', 'adds_encoding',
        'adds_structure', 'adds_text',
        'raises_cognitive', 'lowers_cognitive',
        'requires_', 'demands_', 'lacks_', 'missing_from',
        'absent_making', 'absence_impedes', 'absence_increases',
        'absence_raises', 'insufficient_context',
        'limited_quantity', 'proximity_reduces', 'creates_',
    ]),
]


def classify_predicate(pred):
    for subcat, patterns in PRED_RULES:
        for pat in patterns:
            if pat in pred:
                return subcat
    return 'evaluative'  # safe default


def parse_synset_format(synset_str):
    """'term (category); ...' -> list of (term, category)"""
    import pandas as pd
    if not synset_str or pd.isna(synset_str):
        return []
    pairs = []
    for entry in str(synset_str).split(';'):
        m = re.match(r'(.+?)\s*\((.+?)\)', entry.strip())
        if m:
            pairs.append((m.group(1).strip(), m.group(2).strip()))
    return pairs


# Load data
print("Loading data...")
csv_path = "comment_process/ResultsStepByStep - 4.0.imageDataCompiled.csv"
df = pd.read_csv(csv_path, encoding='cp1252')

json_path = "vc_genome_output_full/three_conditions/oar_B_510.json"
with open(json_path, encoding='utf-8') as f:
    oar_data = json.load(f)

# Build rows
all_rows = []

print("Processing visual_objects...")
for synset_str in df['oar_B_synset_objects'].dropna():
    for term, category in parse_synset_format(synset_str):
        all_rows.append(('visual_objects', category, term))

print("Processing visual_attributes...")
for synset_str in df['oar_B_synset_attributes'].dropna():
    for term, category in parse_synset_format(synset_str):
        all_rows.append(('visual_attributes', category, term))

print("Processing visual_predicates from JSON...")
for img, oar in oar_data.items():
    for rel in oar.get('relationships', []):
        pred = rel.get('pred', '').strip()
        if pred:
            subcat = classify_predicate(pred)
            all_rows.append(('visual_predicates', subcat, pred))

# Aggregate
results = pd.DataFrame(all_rows, columns=['section', 'subcategory', 'term'])
counts = (results
          .groupby(['section', 'subcategory', 'term'])
          .size()
          .reset_index(name='n_images')
          .sort_values(['section', 'subcategory', 'n_images'], ascending=[True, True, False]))

# Output
output_path = "vc_genome/export/oar_B_dict_flat.csv"
counts.to_csv(output_path, index=False)
print(f"\nSaved {len(counts)} rows to {output_path}")

# Summary
print("\n=== SUMMARY ===")
for section in ['visual_objects', 'visual_attributes', 'visual_predicates']:
    sec = counts[counts['section'] == section]
    print(f"\n{section}:")
    for subcat, grp in sec.groupby('subcategory'):
        top = grp.iloc[0]
        print(f"  {subcat:20s}  {len(grp):4d} unique terms  (top: {top['term']}  n={top['n_images']})")
    print(f"  {'TOTAL':20s}  {len(sec):4d} unique terms  across {sec['subcategory'].nunique()} subcategories")
