"""
Shared canonicalization for VC-Genome extractions.

Extracted verbatim from `VCGenome_FullPipeline.ipynb` so that the
human-vs-vision matching analysis reuses the exact same object/predicate
resolution logic as the main pipeline.

Public API:
    resolve_object_synset(name) -> str
    resolve_predicate(pred) -> str
    normalize_attr(attr) -> str
    canonicalize(extraction) -> dict
"""
from __future__ import annotations
import re

OBJECT_SYNSETS = {
    # Marks
    'bar': 'mark.bar', 'bars': 'mark.bar', 'bar_chart': 'mark.bar',
    'stacked_bar': 'mark.bar', 'bar_segment': 'mark.bar',
    'dot': 'mark.point', 'point': 'mark.point', 'data_point': 'mark.point',
    'data_points': 'mark.point', 'dots': 'mark.point', 'scatter_points': 'mark.point',
    'circle': 'mark.point', 'circles': 'mark.point', 'bubble': 'mark.point',
    'shape': 'mark.shape', 'shapes': 'mark.shape',
    'line': 'mark.line', 'lines': 'mark.line', 'vertical_line': 'mark.line',
    'connecting_line': 'mark.line', 'connecting_lines': 'mark.line',
    'curved_lines': 'mark.line', 'dashed_line': 'mark.line',
    'grid_lines': 'mark.line', 'line_curve': 'mark.line',
    'color_segment': 'mark.color_segment', 'color_segments': 'mark.color_segment',
    'color_bands': 'mark.color_segment', 'color_regions': 'mark.color_segment',
    'glyph': 'mark.glyph', 'glyphs': 'mark.glyph',
    'node': 'mark.node', 'nodes': 'mark.node', 'node_cluster': 'mark.node',
    'edge': 'structure.link', 'edges': 'structure.link',
    'area': 'mark.area', 'area_marks': 'mark.area', 'area_shape': 'mark.area',
    'area_chart': 'mark.area', 'stacked_area_chart': 'mark.area',
    'symbol': 'mark.symbol', 'icon': 'mark.symbol', 'arrow': 'mark.symbol',
    'element': 'mark.element', 'object': 'mark.element',
    # Properties
    'color': 'property.color', 'color_shading': 'property.color',
    'color_encoding': 'property.color', 'color_pattern': 'property.color',
    'color_palette': 'property.color', 'color_scheme': 'property.color',
    'color_gradient': 'property.color', 'color_fill': 'property.color',
    'color_scale': 'property.color', 'color_overlay': 'property.color',
    'color_variety': 'property.color',
    'font_size': 'property.font_size',
    'encoding': 'property.encoding', 'visual_encoding': 'property.encoding',
    'data_encoding': 'property.encoding',
    # Structure
    'box': 'structure.panel', 'bounding_box': 'structure.panel',
    'layout': 'structure.layout', 'overall_layout': 'structure.layout',
    'layout_structure': 'structure.layout',
    'pattern': 'structure.pattern', 'data_pattern': 'structure.pattern',
    'flow_pattern': 'structure.pattern',
    'connection': 'structure.link', 'link': 'structure.link',
    'grid': 'structure.grid', 'grid_cells': 'structure.grid',
    'heatmap_grid': 'structure.grid',
    'cell': 'structure.cell',
    'region': 'structure.region', 'data_area': 'structure.region',
    'zoom_inset': 'structure.region',
    'background': 'structure.background', 'map_background': 'structure.background',
    # Text
    'title': 'text.title', 'title_label': 'text.title',
    'description': 'text.description', 'caption': 'text.description',
    'axis_label': 'text.axis_label', 'axis_labels': 'text.axis_label',
    'y_axis_label': 'text.axis_label', 'x_axis_label': 'text.axis_label',
    'label': 'text.label', 'labels': 'text.label', 'text_labels': 'text.label',
    'text_label': 'text.label', 'label_text': 'text.label',
    'node_label': 'text.label', 'node_labels': 'text.label',
    'country_label': 'text.label', 'country_labels': 'text.label',
    'street_labels': 'text.label',
    'text': 'text.label', 'text_content': 'text.label',
    'small_text': 'text.label', 'text_words': 'text.label',
    'large_words': 'text.label', 'explanation_text': 'text.label',
    'words': 'text.label',
    'annotation': 'text.annotation', 'annotation_text': 'text.annotation',
    'annotation_label': 'text.annotation', 'annotation_labels': 'text.annotation',
    'number': 'content.number', 'numbers': 'content.number',
    # Furniture
    'legend': 'furniture.legend', 'color_code': 'furniture.legend',
    'color_legend': 'furniture.legend',
    'axes': 'furniture.axes', 'x_axis': 'furniture.axes', 'y_axis': 'furniture.axes',
    'axis': 'furniture.axes',
    # Whole
    'visualization': 'whole.visualization', 'chart': 'whole.visualization',
    'chart_structure': 'whole.visualization',
    'graph': 'whole.visualization', 'image': 'whole.visualization',
    'map': 'whole.visualization', 'world_map': 'whole.visualization',
    'scatter_plot': 'whole.visualization', 'line_chart': 'whole.visualization',
    'pie_chart': 'whole.visualization', 'word_cloud': 'whole.visualization',
    'node_link_diagram': 'whole.visualization', 'node_link_graph': 'whole.visualization',
    'network_graph': 'whole.visualization',
    # Content
    'data': 'content.data', 'information': 'content.data',
    'knowledge': 'content.domain', 'concept': 'content.domain',
    'detail': 'content.detail',
}

_SUFFIX_MAP = {
    'chart': 'whole.visualization', 'plot': 'whole.visualization',
    'diagram': 'whole.visualization', 'graph': 'whole.visualization',
    'map': 'whole.visualization', 'cloud': 'whole.visualization',
    'legend': 'furniture.legend',
    'axis': 'furniture.axes', 'axes': 'furniture.axes',
    'label': 'text.label', 'labels': 'text.label',
    'text': 'text.label', 'title': 'text.title',
    'annotation': 'text.annotation', 'caption': 'text.description',
    'line': 'mark.line', 'lines': 'mark.line',
    'bar': 'mark.bar', 'bars': 'mark.bar',
    'node': 'mark.node', 'nodes': 'mark.node',
    'point': 'mark.point', 'points': 'mark.point',
    'dot': 'mark.point', 'dots': 'mark.point',
    'circle': 'mark.point', 'circles': 'mark.point',
    'area': 'mark.area',
    'grid': 'structure.grid', 'cell': 'structure.cell', 'cells': 'structure.cell',
    'layout': 'structure.layout', 'region': 'structure.region',
    'background': 'structure.background',
    'color': 'property.color', 'gradient': 'property.color',
    'encoding': 'property.encoding',
    'pattern': 'structure.pattern',
    'link': 'structure.link', 'edge': 'structure.link', 'edges': 'structure.link',
    'data': 'content.data',
}


def resolve_object_synset(name: str) -> str:
    if name is None:
        return 'unknown._none'
    name = name.strip().lower().replace(' ', '_').replace('-', '_')
    if name in OBJECT_SYNSETS:
        return OBJECT_SYNSETS[name]
    parts = name.split('_')
    if len(parts) > 1:
        if parts[-1] in _SUFFIX_MAP:
            return _SUFFIX_MAP[parts[-1]]
        if parts[0] in OBJECT_SYNSETS:
            return OBJECT_SYNSETS[parts[0]]
    if name.endswith('s') and name[:-1] in OBJECT_SYNSETS:
        return OBJECT_SYNSETS[name[:-1]]
    return f'unknown.{name}'


PREDICATE_CANON = {
    'describes': 'describes', 'labels': 'describes', 'annotates': 'describes',
    'clarifies': 'clarifies', 'clarifies_meaning_of': 'clarifies',
    'provides_context_for': 'clarifies',
    'encodes_via': 'encodes_via',
    'fills': 'fills',
    'overlaps_with': 'overlaps_with', 'overlaps': 'overlaps_with',
    'overlap': 'overlaps_with', 'overlays': 'overlaps_with',
    'varies_across': 'varies_across', 'varies_in_size': 'varies_across',
    'connects': 'connects', 'connects_to': 'connects',
    'combines_with': 'co_occurs_with', 'coexists_with': 'co_occurs_with',
    'mixed_with': 'co_occurs_with',
    'clutters': 'clutters', 'clutter': 'clutters',
    'obscures': 'obscures', 'obscures_meaning_of': 'obscures',
    'overwhelms': 'overwhelms',
    'confuses': 'confuses', 'confuse': 'confuses',
    'confuses_interpretation_of': 'confuses',
    'increases': 'increases_complexity',
    'increases_complexity_of': 'increases_complexity',
    'increases_difficulty_of': 'increases_complexity',
    'increases_visual_complexity': 'increases_complexity',
    'adds_complexity_to': 'increases_complexity',
    'increases_processing_effort': 'increases_effort',
    'increases_processing_time': 'increases_effort',
    'increases_reading_effort': 'increases_effort',
    'increases_reading_difficulty': 'increases_effort',
    'increases_interpretive_difficulty': 'increases_effort',
    'increases_interpretive_load_of': 'increases_effort',
    'increases_cognitive_load': 'increases_effort',
    'increases_density_of': 'increases_complexity',
    'increases_ambiguity_of': 'increases_ambiguity',
    'reduces_readability_of': 'hinders_reading',
    'reduces_distinguishability_of': 'hinders_reading',
    'requires_effort_to_read': 'hinders_reading',
    'hinders_interpretation_of': 'hinders_reading',
    'hinders_understanding_of': 'hinders_reading',
    'fails_to_explain': 'fails_to_explain',
    'fails_to_decode': 'fails_to_decode', 'fails_to_clarify': 'fails_to_clarify',
    'lacks_context_for': 'lacks', 'lacks': 'lacks',
    'lacks_story': 'lacks', 'missing_from': 'missing_from',
    'is_missing_from': 'missing_from',
    'aids_interpretation_of': 'aids_interpretation',
    'supports_readability_of': 'aids_interpretation',
    'facilitates_reading_of': 'aids_interpretation',
    'reduces_complexity_of': 'simplifies', 'simplifies': 'simplifies',
    'simplify': 'simplifies',
    'organizes': 'structures', 'structures': 'structures',
    'contains': 'contains', 'contains_many': 'contains',
    'surrounds': 'contains', 'underlies': 'contains',
    'requires_expertise': 'requires_expertise',
    'requires_domain_knowledge': 'requires_expertise',
    'distracts_from': 'distracts',
    'scattered_across': 'distributed_across', 'distributed_across': 'distributed_across',
    'spreads_across': 'distributed_across',
    'differentiates': 'differentiates', 'contrasts_with': 'differentiates',
    'contrasts_against': 'differentiates',
    'represent': 'represents', 'represents': 'represents',
    'show': 'represents', 'display': 'represents', 'forms': 'represents',
    'indicate': 'indicates', 'indicates': 'indicates',
    'suggests_uncertain': 'suggests_uncertain',
    'identify': 'identifies', 'identifies': 'identifies',
    'distinguish': 'distinguishes', 'distinguishes': 'distinguishes',
    'understand': 'interpreted_via', 'interpret': 'interpreted_via',
    'read': 'interpreted_via',
    'compare': 'compared_with', 'compared_with': 'compared_with',
    'more_complex_than': 'compared_with', 'resembles': 'compared_with',
    'contributes_to': 'contributes_to', 'creates': 'contributes_to',
    'makes_hard_to_discern': 'obscures',
}


def resolve_predicate(pred: str) -> str:
    if pred is None:
        return '_none'
    pred = pred.strip().lower().replace(' ', '_').replace('-', '_')
    if pred in PREDICATE_CANON:
        return PREDICATE_CANON[pred]
    for suffix in ('_of', '_for', '_to', '_from'):
        stem = pred.rsplit(suffix, 1)[0]
        if stem in PREDICATE_CANON:
            return PREDICATE_CANON[stem]
    return pred


_WS = re.compile(r'\s+')


def normalize_attr(attr: str) -> str:
    """Light lexical normalization for attribute strings."""
    if attr is None:
        return ''
    s = attr.strip().lower()
    s = s.replace('-', '_').replace(' ', '_')
    s = _WS.sub('_', s)
    return s


def canonicalize(extraction: dict) -> dict:
    """Canonicalize a single {objects, attributes, relationships} extraction."""
    canon = {'objects': [], 'attributes': [], 'relationships': []}
    id_map = {}
    for obj in extraction.get('objects', []):
        synset = resolve_object_synset(obj.get('name', ''))
        id_map[obj['id']] = synset
        canon['objects'].append({
            'id': obj['id'],
            'name': obj.get('name', ''),
            'synset': synset,
            'region': obj.get('region', ''),
        })
    for attr in extraction.get('attributes', []):
        oid = attr.get('object_id')
        if oid not in id_map:
            continue
        canon['attributes'].append({
            'object_id': oid,
            'object_synset': id_map[oid],
            'attr': normalize_attr(attr.get('attr', attr.get('name', ''))),
            'sentiment': attr.get('sentiment', '+'),
            'subtopic': attr.get('subtopic', 'Unknown'),
        })
    for rel in extraction.get('relationships', []):
        s_id, o_id = rel.get('subj'), rel.get('obj')
        if s_id not in id_map or o_id not in id_map:
            continue
        canon['relationships'].append({
            'subject_synset': id_map[s_id],
            'predicate': resolve_predicate(rel.get('pred', '')),
            'object_synset': id_map[o_id],
            'sentiment': rel.get('sentiment', '+'),
            'subtopic': rel.get('subtopic', 'Unknown'),
        })
    return canon
