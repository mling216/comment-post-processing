"""
Render scene graphs for the B_dict condition (9-image test set).
=================================================================
Adapted from render_oar_figure.py — handles the B_dict JSON format:
  - No sentiment field; attributes carry topic + subtopic instead
  - Topic-based color coding for attribute nodes
  - Subtopic shown as secondary label on each attribute node

Outputs:
    vc_genome/scene_graphs/svg/oar_B_dict_<image>.svg
    vc_genome/scene_graphs/png/oar_B_dict_<image>.png

Usage:
    python scripts/render_oar_B_dict.py
"""
from __future__ import annotations
import json, os, re, sys
from collections import defaultdict
from pathlib import Path

import graphviz

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from _vc_canon import resolve_object_synset, resolve_predicate, normalize_attr  # noqa
try:
    from _vc_canon_extended_B import resolve_object_synset_extended as _resolve_ext
except ImportError:
    _resolve_ext = None

IN_JSON   = ROOT / 'vc_genome_output_full' / 'three_conditions' / 'oar_B_dict_9.json'
SVG_DIR   = ROOT / 'vc_genome' / 'scene_graphs' / 'svg'
PNG_DIR   = ROOT / 'vc_genome' / 'scene_graphs' / 'png'
SVG_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR.mkdir(parents=True, exist_ok=True)

# ── Colors ────────────────────────────────────────────────────────────────
COLOR_OBJECT   = '#b8e6b8'   # green  — same as existing conditions
COLOR_RELATION = '#f5b8b8'   # red    — same as existing conditions

TOPIC_COLORS = {
    'Data Density / Image Clutter':           '#fde5b8',   # warm orange
    'Visual Encoding Clarity':                '#b8d8f5',   # sky blue
    'Semantics / Text Legibility':            '#b8f5e8',   # teal
    'Schema':                                 '#e8f5b8',   # yellow-green
    'Color, Symbol, and Texture Details':     '#f5b8e8',   # pink-magenta
    'Aesthetics Uncertainty':                 '#d8d8f5',   # lavender
    'Immediacy / Cognitive Load':             '#f5c7b8',   # salmon
}
DEFAULT_ATTR_COLOR = '#e8e8e8'

# Short labels for the subtopic to keep nodes compact
SUBTOPIC_SHORT: dict[str, str] = {
    'Information Volume':          'Info Vol',
    'Element Quantity':            'Elem Qty',
    'Visual Clutter & Overlap':    'Clutter',
    'Graphical Forms & Primitives':'Graph Forms',
    'Position, Scale & Organization': 'Position',
    'Encoding Interpretability':   'Encoding',
    'Annotations & Labels':        'Labels',
    'Text Volume & Content':       'Text Vol',
    'Typography & Readability':    'Typography',
    'Domain Familiarity':          'Domain',
    'Dimensionality & Structure':  'Dim/Struct',
    'Abstraction Level':           'Abstract',
    'Color Palette & Contrast':    'Color Palette',
    'Symbols & Texture':           'Symbol/Tex',
    'Visual Disorganization':      'Disorg',
    'Perceptual Ambiguity':        'Ambiguity',
    'Interpretive Difficulty':     'Interp Diff',
    'Semantic Clarity':            'Semantic',
    'Processing Time & Effort':    'Effort',
}


# ── Canonicalization (B_dict format — no sentiment) ───────────────────────
def _resolve(name: str) -> str:
    synset = resolve_object_synset(name)
    if synset.startswith('unknown.') and _resolve_ext is not None:
        synset = _resolve_ext(name)
    return synset


def canonicalize_B_dict(raw: dict) -> dict:
    id_to_synset: dict[int, str] = {}
    objects = []
    for o in raw.get('objects', []):
        synset = _resolve(o['name'])
        id_to_synset[o['id']] = synset
        objects.append({
            'id':     o['id'],
            'name':   o['name'],
            'synset': synset,
            'region': o.get('region', 'overall'),
        })

    attributes = []
    for a in raw.get('attributes', []):
        oid    = a['object_id']
        synset = id_to_synset.get(oid, f'unknown.{oid}')
        attributes.append({
            'object_id':     oid,
            'object_synset': synset,
            'attr':          normalize_attr(a.get('attr', '')),
            'topic':         a.get('topic', ''),
            'subtopic':      a.get('subtopic', ''),
        })

    relationships = []
    for r in raw.get('relationships', []):
        subj_synset = id_to_synset.get(r['subj'], f'unknown.{r["subj"]}')
        obj_synset  = id_to_synset.get(r['obj'],  f'unknown.{r["obj"]}')
        relationships.append({
            'subj':           r['subj'],
            'subject_synset': subj_synset,
            'predicate':      resolve_predicate(r.get('pred', '')),
            'obj':            r['obj'],
            'object_synset':  obj_synset,
            'topic':          r.get('topic', ''),
            'subtopic':       r.get('subtopic', ''),
        })

    return {'objects': objects, 'attributes': attributes, 'relationships': relationships}


# ── Graphviz builder ──────────────────────────────────────────────────────
UNDIRECTED_PREDS = {'co_occurs_with', 'overlaps_with'}


def attr_node_label(attr_text: str, subtopic: str) -> str:
    short_sub = SUBTOPIC_SHORT.get(subtopic, subtopic[:12] if subtopic else '')
    label = attr_text.replace('_', ' ')[:35]
    return f'{label}\n[{short_sub}]' if short_sub else label


def build_scene_graph(canon: dict, image_name: str) -> graphviz.Digraph:
    dot = graphviz.Digraph(
        name=image_name,
        format='svg',
        graph_attr={
            'rankdir': 'LR', 'fontsize': '10', 'fontname': 'Arial',
            'nodesep': '0.4', 'ranksep': '0.6', 'splines': 'true',
            'bgcolor': 'white', 'dpi': '150',
        },
        node_attr={
            'fontname': 'Arial', 'fontsize': '10', 'shape': 'box',
            'style': 'rounded,filled', 'penwidth': '1.2',
        },
        edge_attr={'fontname': 'Arial', 'fontsize': '9', 'color': '#888888'},
    )

    # Group attributes by object synset
    attrs_by_obj: dict[str, list] = defaultdict(list)
    for i, attr in enumerate(canon['attributes']):
        label = attr_node_label(attr['attr'], attr.get('subtopic', ''))
        color = TOPIC_COLORS.get(attr.get('topic', ''), DEFAULT_ATTR_COLOR)
        attrs_by_obj[attr['object_synset']].append((f'attr_{i}', label, color))

    # Object nodes
    for obj in canon['objects']:
        dot.node(obj['synset'],
                 label=obj['name'].replace('_', ' '),
                 fillcolor=COLOR_OBJECT, color='#2c3e50',
                 fontcolor='#1a1a1a', penwidth='1.8')

    # Attribute nodes + edges from object
    for obj in canon['objects']:
        for attr_id, attr_label, attr_color in attrs_by_obj.get(obj['synset'], []):
            dot.node(attr_id, label=attr_label,
                     fillcolor=attr_color, color='#7f8c8d', fontcolor='#1a1a1a')
            dot.edge(obj['synset'], attr_id, color='#aaaaaa', arrowsize='0.7')

    # Relationship nodes + edges
    for ri, rel in enumerate(canon['relationships']):
        pred_id = f'rel_{ri}'
        dot.node(pred_id,
                 label=rel['predicate'].replace('_', ' '),
                 fillcolor=COLOR_RELATION, color='#d46a6a', fontcolor='#8b0000')
        dot.edge(rel['subject_synset'], pred_id,
                 color='#d46a6a', penwidth='1.5', arrowsize='0.8')
        if rel['predicate'] in UNDIRECTED_PREDS:
            dot.edge(pred_id, rel['object_synset'],
                     color='#d46a6a', penwidth='1.5',
                     style='dashed', arrowhead='none')
        else:
            dot.edge(pred_id, rel['object_synset'],
                     color='#d46a6a', penwidth='1.5', arrowsize='0.8')

    # Legend + title
    legend_rows = ''.join(
        f'<TD BGCOLOR="{color}" BORDER="1" STYLE="rounded" WIDTH="20"> </TD>'
        f'<TD>{re.sub(r" /.*", "", label)}</TD>'
        for label, color in [
            ('Data Density', '#fde5b8'), ('Visual Encoding', '#b8d8f5'),
            ('Semantics', '#b8f5e8'), ('Schema', '#e8f5b8'),
            ('Color/Symbol', '#f5b8e8'), ('Aesthetics', '#d8d8f5'),
            ('Cognitive Load', '#f5c7b8'),
        ]
    )
    title_html = f'''<
    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2" CELLPADDING="2">
    <TR><TD COLSPAN="16" ALIGN="CENTER">
      <B><FONT POINT-SIZE="13">{image_name} — Condition B_dict</FONT></B>
    </TD></TR>
    <TR>
      <TD BGCOLOR="{COLOR_OBJECT}" BORDER="1" STYLE="rounded" WIDTH="20"> </TD>
      <TD>Object</TD>
      {legend_rows}
      <TD BGCOLOR="{COLOR_RELATION}" BORDER="1" STYLE="rounded" WIDTH="20"> </TD>
      <TD>Relationship</TD>
    </TR>
    </TABLE>>'''
    dot.attr(label=title_html, labelloc='t', labeljust='c')
    return dot


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    if not IN_JSON.exists():
        print(f'ERROR: {IN_JSON} not found — run _extract_oar_B_dict.py first.',
              file=sys.stderr)
        sys.exit(1)

    data = json.loads(IN_JSON.read_text(encoding='utf-8'))
    print(f'Rendering {len(data)} scene graphs …\n')

    _devnull = open(os.devnull, 'w')
    for image_name, raw in data.items():
        if 'error' in raw:
            print(f'  SKIP {image_name}: extraction error')
            continue

        canon = canonicalize_B_dict(raw)
        dot   = build_scene_graph(canon, image_name)
        safe  = image_name.removesuffix('.png')

        _old_stderr = sys.stderr
        sys.stderr  = _devnull
        try:
            svg_bytes = dot.pipe(format='svg')
            png_bytes = dot.pipe(format='png')
        finally:
            sys.stderr = _old_stderr

        (SVG_DIR / f'oar_B_dict_{safe}.svg').write_bytes(svg_bytes)
        (PNG_DIR / f'oar_B_dict_{safe}.png').write_bytes(png_bytes)

        print(f'  {image_name:35s}  '
              f'obj={len(canon["objects"])}  '
              f'attr={len(canon["attributes"])}  '
              f'rel={len(canon["relationships"])}')

    _devnull.close()
    print(f'\nSVGs → {SVG_DIR.relative_to(ROOT.parent)}')
    print(f'PNGs → {PNG_DIR.relative_to(ROOT.parent)}')


if __name__ == '__main__':
    main()
