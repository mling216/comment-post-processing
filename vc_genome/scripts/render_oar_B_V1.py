"""
render_oar_B_V1.py
==================
Render scene graphs for the B and V1 conditions for the same 9 images
as the curate_dict sample, using the same visual style as render_oar_curate_dict.py.

Differences from curate_dict:
  - Source: oar_B_510.json / oar_V1_510.json (filtered to the 9 curate_dict images)
  - Attributes and relationships carry `topic` but no `subtopic`; sentiment is ignored
  - Node second line shows topic abbreviation instead of subtopic abbreviation

Outputs:
  vc_genome/scene_graphs/svg/B/<image>.svg
  vc_genome/scene_graphs/png/B/<image>.png
  vc_genome/scene_graphs/svg/V1/<image>.svg
  vc_genome/scene_graphs/png/V1/<image>.png

Usage:
  python scripts/render_oar_B_V1.py                  # renders both B and V1
  python scripts/render_oar_B_V1.py --condition B
  python scripts/render_oar_B_V1.py --condition V1
"""
from __future__ import annotations
import json, os, sys, argparse
from collections import defaultdict
from pathlib import Path

import graphviz
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
ROOT       = SCRIPT_DIR.parent.parent

TC_DIR       = ROOT / 'vc_genome_output_full' / 'three_conditions'
SAMPLE_JSON  = TC_DIR / 'oar_curate_dict_9.json'   # used only to get the 9 image names
DATA_CSV     = ROOT / 'phrase_reduction_v2' / 'outputs' / 'image_compiled_phrases.csv'

CONDITION_JSON = {
    'B':  TC_DIR / 'oar_B_510.json',
    'V1': TC_DIR / 'oar_V1_510.json',
}

# ── Style (identical to render_oar_curate_dict.py) ────────────────────────
COLOR_OBJECT   = '#b8e6b8'
COLOR_ATTR     = '#b8d4f0'
COLOR_RELATION = '#f5b8b8'
UNDIRECTED_PREDS = {'co_occurs_with', 'overlaps_with'}

TOPIC_ABBREV = {
    'Data Density / Image Clutter':       'Data Density',
    'Visual Encoding Clarity':            'Encoding Clarity',
    'Semantics / Text Legibility':        'Semantics',
    'Schema':                             'Schema',
    'Color, Symbol, and Texture Details': 'Color/Symbol',
    'Aesthetics Uncertainty':             'Aesthetics',
    'Immediacy / Cognitive Load':         'Cogn. Load',
}


def abbrev_topic(topic: str) -> str:
    return TOPIC_ABBREV.get(topic, topic[:18])


# ── Graph builder ─────────────────────────────────────────────────────────

def build_graph(entry: dict, image_name: str, condition: str,
                vis_type: str = '', norm_vc: float | None = None) -> graphviz.Digraph:
    vc_str = f'  VC={norm_vc:.2f}' if norm_vc is not None else ''
    title_line = f'{image_name} — {condition}  [{vis_type}{vc_str}]'

    dot = graphviz.Digraph(
        name=f'{image_name}_{condition}', format='svg',
        graph_attr={
            'rankdir': 'LR', 'fontsize': '10', 'fontname': 'Arial',
            'nodesep': '0.4', 'ranksep': '0.7', 'splines': 'true',
            'bgcolor': 'white', 'dpi': '150',
        },
        node_attr={'fontname': 'Arial', 'fontsize': '10', 'shape': 'box',
                   'style': 'rounded,filled', 'penwidth': '1.2'},
        edge_attr={'fontname': 'Arial', 'fontsize': '9', 'color': '#888888'},
    )

    # Build id → node_id map using LLM-derived names directly
    objects = entry.get('objects', [])
    id_to_node: dict[int, str] = {}
    name_count: dict[str, int] = defaultdict(int)
    for obj in objects:
        name_count[obj['name']] += 1
    name_seen: dict[str, int] = defaultdict(int)
    for obj in objects:
        n = obj['name']
        node_id = f"{n}_{obj['id']}" if name_count[n] > 1 else n
        name_seen[n] += 1
        id_to_node[obj['id']] = node_id
        region = obj.get('region', '')
        region_str = f'\n({region})' if region and region != 'overall' else ''
        dot.node(
            node_id,
            label=f"{n.replace('_', ' ')}{region_str}",
            fillcolor=COLOR_OBJECT,
            color='#2c3e50', fontcolor='#1a1a1a', penwidth='1.8',
        )

    # Attribute nodes (grouped by object)
    attrs_by_obj: dict[int, list] = defaultdict(list)
    for i, attr in enumerate(entry.get('attributes', [])):
        attrs_by_obj[attr['object_id']].append((f'attr_{i}', attr))

    for obj in objects:
        for attr_id, attr in attrs_by_obj.get(obj['id'], []):
            attr_text  = attr['attr'].replace('_', ' ')[:30]
            topic_abbr = abbrev_topic(attr.get('topic', ''))
            label      = f'{attr_text}\n({topic_abbr})'
            dot.node(attr_id, label=label, fillcolor=COLOR_ATTR,
                     color='#5a7fa0', fontcolor='#1a1a1a')
            dot.edge(id_to_node[obj['id']], attr_id, color='#aaaaaa', arrowsize='0.7')

    # Relationship nodes
    for ri, rel in enumerate(entry.get('relationships', [])):
        pred_id    = f'rel_{ri}'
        pred       = rel.get('pred', '')
        pred_text  = pred.replace('_', ' ')
        topic_abbr = abbrev_topic(rel.get('topic', ''))
        label      = f'{pred_text}\n({topic_abbr})'
        subj_node  = id_to_node.get(rel['subj'], str(rel['subj']))
        obj_node   = id_to_node.get(rel['obj'],  str(rel['obj']))
        dot.node(pred_id, label=label, fillcolor=COLOR_RELATION,
                 color='#d46a6a', fontcolor='#8b0000')
        dot.edge(subj_node, pred_id,
                 color='#d46a6a', penwidth='1.5', arrowsize='0.8')
        style     = 'dashed' if pred in UNDIRECTED_PREDS else 'solid'
        arrowhead = 'none'   if pred in UNDIRECTED_PREDS else 'normal'
        dot.edge(pred_id, obj_node,
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
      <TD BGCOLOR="{COLOR_ATTR}"    BORDER="1" STYLE="rounded" WIDTH="20"> </TD><TD>Attribute (topic)</TD>
      <TD BGCOLOR="{COLOR_RELATION}" BORDER="1" STYLE="rounded" WIDTH="20"> </TD><TD>Relationship (topic)</TD>
    </TR>
    </TABLE>>'''
    dot.attr(label=title_html, labelloc='t', labeljust='c')
    return dot


# ── Render one image ──────────────────────────────────────────────────────

def render_image(image_name: str, entry: dict, condition: str,
                 out_svg: Path, out_png: Path,
                 vis_type: str = '', norm_vc: float | None = None):
    dot  = build_graph(entry, image_name, condition,
                       vis_type=vis_type, norm_vc=norm_vc)
    safe = image_name.removesuffix('.png').removesuffix('.jpg')
    _stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        svg_bytes = dot.pipe(format='svg')
        png_bytes = dot.pipe(format='png')
    finally:
        sys.stderr.close()
        sys.stderr = _stderr

    (out_svg / f'{safe}.svg').write_bytes(svg_bytes)
    (out_png / f'{safe}.png').write_bytes(png_bytes)

    n_obj  = len(entry.get('objects', []))
    n_attr = len(entry.get('attributes', []))
    n_rel  = len(entry.get('relationships', []))
    print(f'  {image_name}  [{vis_type}  VC={norm_vc:.2f}]  obj={n_obj}  attr={n_attr}  rel={n_rel}')
    print(f'    → {(out_png / f"{safe}.png").relative_to(ROOT)}')


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', choices=['B', 'V1', 'both'], default='both')
    args = ap.parse_args()

    conditions = ['B', 'V1'] if args.condition == 'both' else [args.condition]

    # Load the 9 target image names from curate_dict sample
    sample_names = list(json.loads(SAMPLE_JSON.read_text(encoding='utf-8')).keys())
    print(f'Target images ({len(sample_names)}): {sample_names}')

    # Load VC metadata
    meta: dict[str, dict] = {}
    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
        for _, row in df.iterrows():
            meta[row['imageName']] = {
                'VisType':      row.get('VisType', ''),
                'NormalizedVC': float(row.get('NormalizedVC', 0)),
            }

    for condition in conditions:
        json_path = CONDITION_JSON[condition]
        all_data  = json.loads(json_path.read_text(encoding='utf-8'))

        out_svg = ROOT / 'vc_genome' / 'scene_graphs' / 'svg' / condition
        out_png = ROOT / 'vc_genome' / 'scene_graphs' / 'png' / condition
        out_svg.mkdir(parents=True, exist_ok=True)
        out_png.mkdir(parents=True, exist_ok=True)

        print(f'\n── Condition: {condition} ──')
        print(f'  SVG → {out_svg.relative_to(ROOT)}')
        print(f'  PNG → {out_png.relative_to(ROOT)}\n')

        missing = [n for n in sample_names if n not in all_data]
        if missing:
            print(f'  WARNING: {len(missing)} images not found in {json_path.name}: {missing}')

        for image_name in sample_names:
            if image_name not in all_data:
                continue
            m = meta.get(image_name, {})
            render_image(
                image_name, all_data[image_name], condition,
                out_svg, out_png,
                vis_type=m.get('VisType', ''),
                norm_vc=m.get('NormalizedVC'),
            )


if __name__ == '__main__':
    main()
