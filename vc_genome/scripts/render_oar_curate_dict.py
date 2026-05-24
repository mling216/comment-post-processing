"""
render_oar_curate_dict.py
=========================
Render scene graphs for all images in oar_curate_dict_9.json (or any
curate_dict output) using the same graphviz style as the main pipeline.

Differences from B/V1/V2 renderers:
  - No sentiment field → all attribute nodes use a single neutral colour
  - Subtopic shown in smaller text inside each attribute node
  - Title includes VisType and NormalizedVC

Outputs (one file per image per format):
  vc_genome/scene_graphs/svg/oar_<image>_curate_dict.svg
  vc_genome/scene_graphs/png/oar_<image>_curate_dict.png

Usage:
  python scripts/render_oar_curate_dict.py
  python scripts/render_oar_curate_dict.py --input vc_genome_output_full/three_conditions/oar_curate_dict.json
"""
from __future__ import annotations
import json, os, sys, argparse
from collections import defaultdict
from pathlib import Path

import graphviz
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
ROOT       = SCRIPT_DIR.parent.parent   # vc_genome/scripts/ → vc_genome/ → comment_post_processing/
sys.path.insert(0, str(SCRIPT_DIR))
from _vc_canon import canonicalize  # noqa

# ── Directories ───────────────────────────────────────────────────────────
DEFAULT_IN  = ROOT / 'vc_genome_output_full' / 'three_conditions' / 'oar_curate_dict_9.json'
OUT_SVG     = ROOT / 'vc_genome' / 'scene_graphs' / 'svg' / 'curated_dict'
OUT_PNG     = ROOT / 'vc_genome' / 'scene_graphs' / 'png' / 'curated_dict'
DATA_CSV    = ROOT / 'phrase_reduction_v2' / 'outputs' / 'image_compiled_phrases.csv'

OUT_SVG.mkdir(parents=True, exist_ok=True)
OUT_PNG.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────
COLOR_OBJECT   = '#b8e6b8'   # green  — visual objects
COLOR_ATTR     = '#b8d4f0'   # blue   — attributes (no sentiment distinction)
COLOR_RELATION = '#f5b8b8'   # red    — relationship predicates
UNDIRECTED_PREDS = {'co_occurs_with', 'overlaps_with'}

SUBTOPIC_ABBREV = {
    'Information Volume':            'Info Volume',
    'Element Quantity':              'Elem Qty',
    'Visual Clutter & Overlap':      'Clutter/Overlap',
    'Graphical Forms & Primitives':  'Forms & Prims',
    'Position, Scale & Organization':'Position/Scale',
    'Encoding Interpretability':     'Encoding Interp.',
    'Annotations & Labels':          'Annot. & Labels',
    'Text Volume & Content':         'Text Volume',
    'Typography & Readability':      'Typography',
    'Domain Familiarity':            'Domain Know.',
    'Dimensionality & Structure':    'Dim. & Structure',
    'Abstraction Level':             'Abstraction',
    'Color Palette & Contrast':      'Color Palette',
    'Symbols & Texture':             'Symbols',
    'Visual Disorganization':        'Disorganization',
    'Perceptual Ambiguity':          'Percept. Ambig.',
    'Interpretive Difficulty':       'Interp. Diff.',
    'Semantic Clarity':              'Semantic Clarity',
    'Processing Time & Effort':      'Proc. Effort',
}


def abbrev_subtopic(st: str) -> str:
    return SUBTOPIC_ABBREV.get(st, st[:18])


# ── Graph builder ─────────────────────────────────────────────────────────

def build_graph(canon: dict, image_name: str,
                vis_type: str = '', norm_vc: float | None = None) -> graphviz.Digraph:
    vc_str = f'  VC={norm_vc:.2f}' if norm_vc is not None else ''
    title_line = f'{image_name} — curate_dict  [{vis_type}{vc_str}]'

    dot = graphviz.Digraph(
        name=f'{image_name}_curate_dict', format='svg',
        graph_attr={
            'rankdir':  'LR',
            'fontsize': '10',
            'fontname': 'Arial',
            'nodesep':  '0.4',
            'ranksep':  '0.7',
            'splines':  'true',
            'bgcolor':  'white',
            'dpi':      '150',
        },
        node_attr={'fontname': 'Arial', 'fontsize': '10', 'shape': 'box',
                   'style': 'rounded,filled', 'penwidth': '1.2'},
        edge_attr={'fontname': 'Arial', 'fontsize': '9', 'color': '#888888'},
    )

    # Object nodes
    for obj in canon['objects']:
        region = obj.get('region', '')
        region_str = f'\n({region})' if region and region != 'overall' else ''
        dot.node(
            obj['synset'],
            label=f"{obj['name'].replace('_', ' ')}{region_str}",
            fillcolor=COLOR_OBJECT,
            color='#2c3e50',
            fontcolor='#1a1a1a',
            penwidth='1.8',
        )

    # Attribute nodes (grouped by object)
    attrs_by_obj = defaultdict(list)
    for i, attr in enumerate(canon['attributes']):
        attrs_by_obj[attr['object_synset']].append((f'attr_{i}', attr))

    for obj in canon['objects']:
        for attr_id, attr in attrs_by_obj.get(obj['synset'], []):
            attr_text  = attr['attr'].replace('_', ' ')[:30]
            sub_abbrev = abbrev_subtopic(attr.get('subtopic', ''))
            label      = f'{attr_text}\n({sub_abbrev})'
            dot.node(attr_id, label=label, fillcolor=COLOR_ATTR,
                     color='#5a7fa0', fontcolor='#1a1a1a')
            dot.edge(obj['synset'], attr_id, color='#aaaaaa', arrowsize='0.7')

    # Relationship nodes
    for ri, rel in enumerate(canon['relationships']):
        pred_id    = f'rel_{ri}'
        pred_text  = rel['predicate'].replace('_', ' ')
        sub_abbrev = abbrev_subtopic(rel.get('subtopic', ''))
        label      = f'{pred_text}\n({sub_abbrev})'
        dot.node(pred_id, label=label, fillcolor=COLOR_RELATION,
                 color='#d46a6a', fontcolor='#8b0000')
        dot.edge(rel['subject_synset'], pred_id,
                 color='#d46a6a', penwidth='1.5', arrowsize='0.8')
        style = 'dashed' if rel['predicate'] in UNDIRECTED_PREDS else 'solid'
        arrowhead = 'none' if rel['predicate'] in UNDIRECTED_PREDS else 'normal'
        dot.edge(pred_id, rel['object_synset'],
                 color='#d46a6a', penwidth='1.5',
                 style=style, arrowhead=arrowhead)

    # Legend + title
    title_html = f'''<
    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2" CELLPADDING="2">
    <TR><TD COLSPAN="6" ALIGN="CENTER">
      <B><FONT POINT-SIZE="13">{title_line}</FONT></B>
    </TD></TR>
    <TR>
      <TD BGCOLOR="{COLOR_OBJECT}"  BORDER="1" STYLE="rounded" WIDTH="20"> </TD><TD>Object (region)</TD>
      <TD BGCOLOR="{COLOR_ATTR}"    BORDER="1" STYLE="rounded" WIDTH="20"> </TD><TD>Attribute (subtopic)</TD>
      <TD BGCOLOR="{COLOR_RELATION}" BORDER="1" STYLE="rounded" WIDTH="20"> </TD><TD>Relationship (subtopic)</TD>
    </TR>
    </TABLE>>'''
    dot.attr(label=title_html, labelloc='t', labeljust='c')
    return dot


# ── Render one image ──────────────────────────────────────────────────────

def render_image(image_name: str, extraction: dict,
                 vis_type: str = '', norm_vc: float | None = None):
    canon = canonicalize(extraction)
    dot   = build_graph(canon, image_name, vis_type=vis_type, norm_vc=norm_vc)

    safe = image_name.removesuffix('.png').removesuffix('.jpg')
    # Suppress graphviz stderr
    _stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        svg_bytes = dot.pipe(format='svg')
        png_bytes = dot.pipe(format='png')
    finally:
        sys.stderr.close()
        sys.stderr = _stderr

    svg_path = OUT_SVG / f'{safe}.svg'
    png_path = OUT_PNG / f'{safe}.png'
    svg_path.write_bytes(svg_bytes)
    png_path.write_bytes(png_bytes)

    print(f'  {image_name}  [{vis_type}  VC={norm_vc:.2f}]  '
          f'obj={len(canon["objects"])}  '
          f'attr={len(canon["attributes"])}  '
          f'rel={len(canon["relationships"])}')
    print(f'    → {png_path.relative_to(ROOT)}')


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Render curate_dict scene graphs')
    ap.add_argument('--input', type=str, default=str(DEFAULT_IN),
                    help='Path to curate_dict JSON file')
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f'Input not found: {in_path}')

    data = json.loads(in_path.read_text(encoding='utf-8'))
    print(f'Loaded {len(data)} images from {in_path.name}')

    # Load VC metadata for titles
    meta: dict[str, dict] = {}
    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
        for _, row in df.iterrows():
            meta[row['imageName']] = {
                'VisType':      row.get('VisType', ''),
                'NormalizedVC': float(row.get('NormalizedVC', 0)),
            }

    print(f'\nRendering to:')
    print(f'  SVG → {OUT_SVG.relative_to(ROOT)}')
    print(f'  PNG → {OUT_PNG.relative_to(ROOT)}')
    print()

    for image_name, extraction in sorted(data.items()):
        m = meta.get(image_name, {})
        render_image(
            image_name, extraction,
            vis_type=m.get('VisType', ''),
            norm_vc=m.get('NormalizedVC'),
        )

    print(f'\nDone. {len(data)} scene graphs written.')


if __name__ == '__main__':
    main()
