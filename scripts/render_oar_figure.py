"""
Render side-by-side scene graphs for one image — Condition B (human-phrase
grounded) and Condition V2 (vision + B-sourced anchors) — in the same
graphviz style as `VCGenome_FullPipeline.ipynb`'s `build_vg_graphviz`.

Outputs (paper/figures/):
    oar_<image>_B.{png,svg}
    oar_<image>_V2.{png,svg}
"""
from __future__ import annotations
import json, os, sys
from collections import defaultdict
from pathlib import Path
import graphviz

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from _vc_canon import canonicalize  # noqa

IMAGE = 'SciVisJ.2926.1.png'

IN_DIR  = ROOT / 'vc_genome_output_full' / 'three_conditions'
OUT_DIR = ROOT / 'paper' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Style (mirrors VCGenome_FullPipeline.ipynb build_vg_graphviz) ────────
COLOR_OBJECT    = '#b8e6b8'
COLOR_ATTR_POS  = '#c7b8e6'
COLOR_ATTR_NEG  = '#e8dff5'
COLOR_RELATION  = '#f5b8b8'
UNDIRECTED_PREDS = {'co_occurs_with', 'overlaps_with'}


def build_vg_graphviz(canon_data, image_name, condition_label):
    dot = graphviz.Digraph(
        name=f'{image_name}_{condition_label}', format='svg',
        graph_attr={'rankdir': 'LR', 'fontsize': '10', 'fontname': 'Arial',
                    'nodesep': '0.4', 'ranksep': '0.6', 'splines': 'true',
                    'bgcolor': 'white', 'dpi': '150'},
        node_attr={'fontname': 'Arial', 'fontsize': '10', 'shape': 'box',
                   'style': 'rounded,filled', 'penwidth': '1.2'},
        edge_attr={'fontname': 'Arial', 'fontsize': '9', 'color': '#888888'},
    )

    attrs_by_obj = defaultdict(list)
    for i, attr in enumerate(canon_data['attributes']):
        sent  = '(+)' if attr['sentiment'] == '+' else '(\u2212)'
        label = f"{sent} {attr['attr'].replace('_', ' ')[:30]}"
        color = COLOR_ATTR_POS if attr['sentiment'] == '+' else COLOR_ATTR_NEG
        attrs_by_obj[attr['object_synset']].append((f'attr_{i}', label, color))

    for obj in canon_data['objects']:
        dot.node(obj['synset'], label=obj['name'].replace('_', ' '),
                 fillcolor=COLOR_OBJECT, color='#2c3e50',
                 fontcolor='#1a1a1a', penwidth='1.8')
    for obj in canon_data['objects']:
        for attr_id, attr_label, attr_color in attrs_by_obj.get(obj['synset'], []):
            dot.node(attr_id, label=attr_label, fillcolor=attr_color,
                     color='#7f8c8d', fontcolor='#1a1a1a')
            dot.edge(obj['synset'], attr_id, color='#aaaaaa', arrowsize='0.7')

    for ri, rel in enumerate(canon_data['relationships']):
        pred_id = f'rel_{ri}'
        dot.node(pred_id, label=rel['predicate'].replace('_', ' '),
                 fillcolor=COLOR_RELATION, color='#d46a6a',
                 fontcolor='#8b0000')
        dot.edge(rel['subject_synset'], pred_id, color='#d46a6a',
                 penwidth='1.5', arrowsize='0.8')
        if rel['predicate'] in UNDIRECTED_PREDS:
            dot.edge(pred_id, rel['object_synset'], color='#d46a6a',
                     penwidth='1.5', style='dashed', arrowhead='none')
        else:
            dot.edge(pred_id, rel['object_synset'], color='#d46a6a',
                     penwidth='1.5', arrowsize='0.8')

    title_html = f'''<
    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2" CELLPADDING="2">
    <TR><TD COLSPAN="8" ALIGN="CENTER"><B><FONT POINT-SIZE="14">{image_name} \u2014 Condition {condition_label}</FONT></B></TD></TR>
    <TR>
      <TD BGCOLOR="#b8e6b8" BORDER="1" STYLE="rounded" WIDTH="20"> </TD><TD>Object</TD>
      <TD BGCOLOR="#c7b8e6" BORDER="1" STYLE="rounded" WIDTH="20"> </TD><TD>Attr (+)</TD>
      <TD BGCOLOR="#e8dff5" BORDER="1" STYLE="rounded" WIDTH="20"> </TD><TD>Attr (\u2212)</TD>
      <TD BGCOLOR="#f5b8b8" BORDER="1" STYLE="rounded" WIDTH="20"> </TD><TD>Relationship</TD>
    </TR>
    </TABLE>>'''
    dot.attr(label=title_html, labelloc='t', labeljust='c')
    return dot


def render(condition_label: str, raw_extraction: dict):
    canon = canonicalize(raw_extraction)
    dot = build_vg_graphviz(canon, IMAGE, condition_label)

    safe = IMAGE.removesuffix('.png')
    _stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        svg_bytes = dot.pipe(format='svg')
        png_bytes = dot.pipe(format='png')
        pdf_bytes = dot.pipe(format='pdf')
    finally:
        sys.stderr.close()
        sys.stderr = _stderr

    (OUT_DIR / f'oar_{safe}_{condition_label}.svg').write_bytes(svg_bytes)
    (OUT_DIR / f'oar_{safe}_{condition_label}.png').write_bytes(png_bytes)
    (OUT_DIR / f'oar_{safe}_{condition_label}.pdf').write_bytes(pdf_bytes)
    print(f'  [{condition_label}]  obj={len(canon["objects"])}  '
          f'attr={len(canon["attributes"])}  rel={len(canon["relationships"])}'
          f'  -> oar_{safe}_{condition_label}.png/.svg/.pdf')


def main():
    oar_B  = json.loads((IN_DIR / 'oar_B.json').read_text(encoding='utf-8'))
    oar_V2 = json.loads((IN_DIR / 'oar_V2.json').read_text(encoding='utf-8'))
    print(f'Image: {IMAGE}')
    render('B',  oar_B[IMAGE])
    render('V2', oar_V2[IMAGE])


if __name__ == '__main__':
    main()
