"""
Build oar_B_dict_flat_raw.csv — same three-level structure as oar_B_dict_flat.csv
but using RAW object/attribute/predicate names from oar_B_510.json instead of
the post-hoc synset.

Sections:
  visual_objects     subcategories: chart_type, scaffold, encoding, mark, text, layout, content
  visual_attributes  subcategories: from JSON topic field (normalised)
  visual_predicates  subcategories: spatial, semantic, structural, perceptual, cognitive, evaluative

Outputs: vc_genome/export/oar_B_dict_flat_raw.csv
"""

import re
import json
import pandas as pd

# ── Object classification rules ───────────────────────────────────────────────
# Priority-ordered; each entry = (subcategory, [word-level stems to match]).
# Matching is done on the set of underscore-split words in the name.
OBJ_RULES = [
    # SCAFFOLD – structural navigation frame of the chart
    ('scaffold', {
        'legend', 'axis', 'axes', 'grid', 'scale', 'color_bar', 'color_key',
        'color_scale', 'legend_key', 'scale_lines', 'numeric_scale',
        'measurement_bar', 'color_range', 'longitudinal_axis',
        'horizontal_axis', 'x_axis', 'y_axis', 'time_axis', 'timeline',
    }),
    # ENCODING – visual encoding channels (colour, size, texture, font)
    ('encoding', {
        'color_scheme', 'color_encoding', 'color_palette', 'color_gradient',
        'color_fill', 'color_coding', 'color_overlay', 'color_shading',
        'color_contrast', 'color_distribution', 'color_segment',
        'color_segments', 'color_code', 'color_codes', 'color_contrasts',
        'color_band', 'color_bands', 'color_shades',
        'colors', 'color',
        'blue_colors', 'red_colors', 'white_color',
        'colored_segments', 'colored_surface',
        'visual_encoding', 'size_encoding', 'shape_encoding',
        'data_encoding', 'encoding_scheme', 'draw_style',
        'shading', 'gradient', 'font_styling', 'font', 'typography',
        'calligraphy', 'texture', 'half_shading', 'striped_pattern',
    }),
    # CHART_TYPE – the overall kind of visualisation
    ('chart_type', {
        'chart', 'visualization', 'map', 'graph', 'diagram', 'heatmap',
        'word_cloud', 'scatter', 'histogram', 'matrix', 'network',
        'phylogenetic', 'bar_graph', 'line_chart', 'pie_chart',
        'circle_graph', 'heat_graph', 'map_graph', 'weight_graph',
        'wind_map', 'weather_map', 'geographical_map', 'data_visualization',
        'visual_display', '3d_visualization', 'data_visuali', 'treemap',
        'multi_panel_figure', 'chart_collection', 'charts',
        'table', 'plot', 'overall_image', 'overall_visualization',
    }),
    # MARK – visual marks / data glyphs
    ('mark', {
        'line', 'lines', 'dot', 'dots', 'circle', 'circles', 'square',
        'squares', 'bar', 'bars', 'shape', 'shapes', 'point', 'points',
        'mark', 'marks', 'symbol', 'symbols', 'arrow', 'arrows', 'node',
        'nodes', 'edge', 'edges', 'path', 'paths', 'curve', 'spiral',
        'spirals', 'box', 'boxes', 'stick', 'sticks', 'pixel', 'pixels',
        'spot', 'spots', 'intersection', 'intersections', 'cross', 'icon',
        'icons', 'element', 'elements', 'data_points', 'data_elements',
        'data_marks', 'data_lines', 'data_items', 'data_entities',
        'data_segments', 'visual_elements', 'visual_marks', 'visual_items',
        'visual_cues', 'graphic_marks', 'ink_marks', 'point_shapes',
        'dot_marks', 'plotted_points', 'overlapping_lines', 'dashed_line',
        'dashed_box', 'trend_line', 'error_bars', 'data_marks',
        'filled_bars', 'rectangular_shapes', 'circular_shapes',
        'geometric_shapes', 'abstract_shapes', 'animal_shapes',
        'country_shapes', 'graph_metrics', 'chart_shapes', 'connections',
        'crossing_lines', 'horizontal_lines', 'vertical_lines',
        'curved_lines', 'dotted_lines', 'straight_line', 'curve_lines',
        'scale_lines', 'interconnecting_lines', 'chart_elements',
        'graphic_elements', 'visual_element', 'graphic_element',
        'graphical_marks', 'pixel_elements', 'data_figures',
        '3d_elements', 'box_indicators', 'meeting_points', 'motion',
        'branches', 'cubes', 'seismic_waveform', 'waterfall_pattern',
        'flow_pattern', 'link',
    }),
    # TEXT – labels, words, annotations, text blocks
    ('text', {
        'text_labels', 'labels', 'text_elements', 'words', 'text',
        'letters', 'annotation', 'annotations', 'description', 'captions',
        'title', 'text_content', 'text_display', 'text_words', 'text_items',
        'numerical_labels', 'country_labels', 'species_labels', 'brand_labels',
        'axis_labels', 'date_labels', 'numeric_labels', 'chart_titles',
        'highlighted_words', 'large_words', 'small_words', 'purpose_label',
        'bottom_text', 'description_text', 'explanation_text',
        'labels_annotations', 'codes_numbers', 'labels_values',
        'criteria_labels', 'chemical_compound_labels', 'title_annotation',
        'title_or_label', 'labels_text', 'axes_titles', 'small_text',
        'labels_or_legend', 'text_label', 'text_box', 'text_boxes',
        'numerical_values', 'numeric_values', 'formula_symbol',
        'mathematical_formulae', 'subject_label', 'terminology',
    }),
    # LAYOUT – spatial arrangement, panels, background
    ('layout', {
        'layout', 'background', 'white_space', 'negative_space', 'structure',
        'panel', 'region', 'area', 'sections', 'columns', 'rows', 'frame',
        'division', 'divisions', 'layers', 'presentation_logic',
        'chart_layout', 'visual_layout', 'layout_structure', 'chart_area',
        'chart_structure', 'hierarchy', 'hierarchy_levels',
        'hierarchy_structure', 'hierarchical_structure',
        'individual_panel', 'image_panels', 'subplots', 'left_chart',
        'right_chart', 'graph_panel_left', 'graph_panel_right',
        'left_image', 'zoom_area', 'magnification_inset',
        'columns_and_rows', 'dimension_matrix', 'empty_spaces',
    }),
    # CONTENT – data, information, numeric content
    ('content', {
        'data', 'data_content', 'data_area', 'data_values', 'data_details',
        'information', 'information_content', 'information_elements',
        'information_noise', 'numbers', 'number', 'variables', 'measurements',
        'values', 'statistics', 'metrics', 'metric_representation',
        'quantity', 'parameters', 'categories', 'category', 'groups',
        'entities', 'subject_matter', 'content', 'distribution',
        'pattern', 'key_patterns', 'patterns', 'details', 'small_details',
        'detail_features', 'context', 'semantic_content', 'word_frequency',
        'biological_content', 'chemical_data', 'wavelet_coefficients',
        'field_information', 'relationships', 'interactions', 'ratio',
        'dataset', 'dataset_1', 'dataset_2', 'data_subjects',
        'data_types', 'data_pattern', 'data_entries', 'time_period',
        'spatial_dimension', 'crop_type',
    }),
]

# Build priority-respecting exact lookup: first rule wins (do NOT overwrite)
_OBJ_EXACT = {}  # exact full-name match -> subcategory
for subcat, names in OBJ_RULES:
    for n in names:
        if n not in _OBJ_EXACT:   # first match wins — preserves priority order
            _OBJ_EXACT[n] = subcat


def classify_object(name: str) -> str:
    """Classify a raw object name into a subcategory."""
    # Exact match first
    if name in _OBJ_EXACT:
        return _OBJ_EXACT[name]
    # Partial: check if name contains any key token (priority order)
    for subcat, names in OBJ_RULES:
        for token in names:
            if token in name:
                return subcat
    return 'content'  # safe fallback


# ── Predicate classification rules (unchanged from synset version) ────────────
PRED_RULES = [
    ('spatial', [
        'overlap', 'intersect', 'scatter', 'distribut', 'plotted',
        'arranged_along', 'arranged_in', 'aligned_with',
        'converge', 'cluster', 'bounded_by', 'extends_along', 'packed',
        'span', 'transition', 'embedded', 'fills', 'paired_with',
        'connected_by', 'branches_into', 'too_close', 'part_of',
        'underlies', 'repeated_along', 'shares_time_axis',
        'displayed_on', 'positioned',
    ]),
    ('semantic', [
        'encod', 'represent', 'annotate', 'annotated',
        'communicat', 'convey', 'mapped_to', 'supplement',
        'summariz', 'explains', 'labeled_by', 'labeled_with',
        'labels', 'does_not_encode', 'ambiguously_encodes',
    ]),
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


def classify_predicate(pred: str) -> str:
    for subcat, patterns in PRED_RULES:
        for pat in patterns:
            if pat in pred:
                return subcat
    return 'evaluative'


# ── Topic normalisation for attributes ───────────────────────────────────────
TOPIC_MAP = {
    'Color, Symbol, and Texture Details': 'color_symbol_texture',
    'Data Density / Image Clutter':        'data_density_clutter',
    'Immediacy / Cognitive Load':          'cognitive_load',
    'Schema':                              'schema_familiarity',
    'Semantics / Text Legibility':         'text_legibility',
    'Visual Encoding Clarity':             'encoding_clarity',
    'Aesthetics Uncertainty':              'aesthetics_uncertainty',
    'Spatial Organization':                'spatial_organization',
    'Chart Type Familiarity':              'chart_familiarity',
    'Overall Complexity':                  'overall_complexity',
}


def normalise_topic(topic: str) -> str:
    if not topic:
        return 'other'
    t = topic.strip()
    if t in TOPIC_MAP:
        return TOPIC_MAP[t]
    # Generic normalisation: lowercase, replace punctuation/spaces
    return re.sub(r'[^a-z0-9]+', '_', t.lower()).strip('_')


# ── Load JSON ─────────────────────────────────────────────────────────────────
print("Loading JSON...")
json_path = "vc_genome_output_full/three_conditions/oar_B_510.json"
with open(json_path, encoding='utf-8') as f:
    oar_data = json.load(f)

# ── Build rows ────────────────────────────────────────────────────────────────
all_rows = []

print("Processing visual_objects (raw)...")
for img, oar in oar_data.items():
    id_to_name = {o['id']: o.get('name', '').strip()
                  for o in oar.get('objects', [])}
    for obj in oar.get('objects', []):
        name = obj.get('name', '').strip()
        if name:
            subcat = classify_object(name)
            all_rows.append(('visual_objects', subcat, name))

print("Processing visual_attributes (raw by topic)...")
for img, oar in oar_data.items():
    for attr in oar.get('attributes', []):
        a = attr.get('attr', '').strip()
        topic = normalise_topic(attr.get('topic', ''))
        if a:
            all_rows.append(('visual_attributes', topic, a))

print("Processing visual_predicates (raw)...")
for img, oar in oar_data.items():
    for rel in oar.get('relationships', []):
        pred = rel.get('pred', '').strip()
        if pred:
            subcat = classify_predicate(pred)
            all_rows.append(('visual_predicates', subcat, pred))

# ── Aggregate ─────────────────────────────────────────────────────────────────
results = pd.DataFrame(all_rows, columns=['section', 'subcategory', 'term'])
counts = (results
          .groupby(['section', 'subcategory', 'term'])
          .size()
          .reset_index(name='n_images')
          .sort_values(['section', 'subcategory', 'n_images'], ascending=[True, True, False]))

# ── Output ────────────────────────────────────────────────────────────────────
output_path = "vc_genome/export/oar_B_dict_flat_raw.csv"
counts.to_csv(output_path, index=False)
print(f"\nSaved {len(counts)} rows to {output_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== SUMMARY ===")
for section in ['visual_objects', 'visual_attributes', 'visual_predicates']:
    sec = counts[counts['section'] == section]
    print(f"\n{section}:")
    for subcat, grp in sec.groupby('subcategory'):
        top = grp.iloc[0]
        print(f"  {subcat:30s}  {len(grp):4d} unique terms  "
              f"(top: {top['term']}  n={top['n_images']})")
    print(f"  {'TOTAL':30s}  {len(sec):4d} unique terms  "
          f"across {sec['subcategory'].nunique()} subcategories")
