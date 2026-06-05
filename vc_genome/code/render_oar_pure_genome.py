"""
render_oar_pure_genome.py
=========================
Render scene graphs for the pure-genome condition.

Input:  vc_genome/export/oar_pure_genome_9.json  (or --input for any file)
Output: vc_genome/scene_graphs/svg/pure_genome/<image>.svg
        vc_genome/scene_graphs/png/pure_genome/<image>.png

Differences from curate_dict / B / V1 renderers:
  - No topic or subtopic — attribute and relationship nodes show text only
  - Legend reads "Attribute" and "Relationship" (no topic/subtopic annotation)
  - Metadata loaded from ResultsStepByStep_4.0.imageDataCompiled.csv

Usage:
  python code/render_oar_pure_genome.py
  python code/render_oar_pure_genome.py --input vc_genome/export/oar_pure_genome_full.json
"""
from __future__ import annotations
import json, os, sys, argparse
from collections import defaultdict
from pathlib import Path

import graphviz
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
ROOT       = SCRIPT_DIR.parent.parent   # comment_post_processing/

DEFAULT_IN  = ROOT / 'vc_genome' / 'export' / 'oar_pure_genome_full.json'
OUT_SVG     = ROOT / 'vc_genome' / 'scene_graphs' / 'svg' / 'pure_genome'
OUT_PNG     = ROOT / 'vc_genome' / 'scene_graphs' / 'png' / 'pure_genome'
DATA_CSV    = ROOT / 'comment_process' / 'ResultsStepByStep - 4.0.imageDataCompiled.csv'

# ── Style (same palette as all other condition renderers) ─────────────────
COLOR_OBJECT   = '#b8e6b8'   # green
COLOR_ATTR     = '#b8d4f0'   # blue
COLOR_RELATION = '#f5b8b8'   # red
UNDIRECTED_PREDS = {'co_occurs_with', 'overlaps_with'}


# ── Graph builder ─────────────────────────────────────────────────────────

def build_graph(entry: dict, image_name: str,
                vis_type: str = '', norm_vc: float | None = None) -> graphviz.Digraph:
    vc_str     = f'  VC={norm_vc:.2f}' if norm_vc is not None else ''
    title_line = f'{image_name} — pure-genome  [{vis_type}{vc_str}]'

    dot = graphviz.Digraph(
        name=f'{image_name}_pure_genome', format='svg',
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

    # Object nodes — id → node_id, disambiguate duplicate names
    objects = entry.get('objects', [])
    name_count: dict[str, int] = defaultdict(int)
    for obj in objects:
        name_count[obj['name']] += 1

    # Object IDs that are referenced by at least one attribute or relationship
    referenced_obj_ids = {a['object_id'] for a in entry.get('attributes', [])}
    for rel in entry.get('relationships', []):
        referenced_obj_ids.add(rel['subj'])
        referenced_obj_ids.add(rel['obj'])

    id_to_node: dict[int, str] = {}
    for obj in objects:
        n       = obj['name']
        node_id = f"{n}_{obj['id']}" if name_count[n] > 1 else n
        id_to_node[obj['id']] = node_id
        # Skip id=0 anchor if nothing connects to it (isolated node)
        if obj['id'] == 0 and 0 not in referenced_obj_ids:
            continue
        region = obj.get('region', '')
        region_str = f'\n({region})' if region and region != 'overall' else ''
        dot.node(
            node_id,
            label=f"{n.replace('_', ' ')}{region_str}",
            fillcolor=COLOR_OBJECT,
            color='#2c3e50', fontcolor='#1a1a1a', penwidth='1.8',
        )

    # Attribute nodes (no topic/subtopic second line)
    attrs_by_obj: dict[int, list] = defaultdict(list)
    for i, attr in enumerate(entry.get('attributes', [])):
        attrs_by_obj[attr['object_id']].append((f'attr_{i}', attr))

    for obj in objects:
        for attr_id, attr in attrs_by_obj.get(obj['id'], []):
            raw_text = attr.get('attr') or attr.get('name') or ''
            if not raw_text:
                continue
            attr_text = raw_text.replace('_', ' ')[:35]
            dot.node(attr_id, label=attr_text,
                     fillcolor=COLOR_ATTR, color='#5a7fa0', fontcolor='#1a1a1a')
            dot.edge(id_to_node[obj['id']], attr_id,
                     color='#aaaaaa', arrowsize='0.7')

    # Relationship nodes (no topic/subtopic second line)
    for ri, rel in enumerate(entry.get('relationships', [])):
        pred_id   = f'rel_{ri}'
        pred      = rel.get('pred', '')
        pred_text = pred.replace('_', ' ')
        subj_node = id_to_node.get(rel['subj'], str(rel['subj']))
        obj_node  = id_to_node.get(rel['obj'],  str(rel['obj']))
        dot.node(pred_id, label=pred_text, fillcolor=COLOR_RELATION,
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
      <TD BGCOLOR="{COLOR_OBJECT}"   BORDER="1" STYLE="rounded" WIDTH="20"> </TD><TD>Object (region)</TD>
      <TD BGCOLOR="{COLOR_ATTR}"     BORDER="1" STYLE="rounded" WIDTH="20"> </TD><TD>Attribute</TD>
      <TD BGCOLOR="{COLOR_RELATION}" BORDER="1" STYLE="rounded" WIDTH="20"> </TD><TD>Relationship</TD>
    </TR>
    </TABLE>>'''
    dot.attr(label=title_html, labelloc='t', labeljust='c')
    return dot


# ── Render one image ──────────────────────────────────────────────────────

def render_image(image_name: str, entry: dict,
                 vis_type: str = '', norm_vc: float | None = None):
    dot  = build_graph(entry, image_name, vis_type=vis_type, norm_vc=norm_vc)
    safe = image_name.removesuffix('.png').removesuffix('.jpg')

    _stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        svg_bytes = dot.pipe(format='svg')
        png_bytes = dot.pipe(format='png')
    finally:
        sys.stderr.close()
        sys.stderr = _stderr

    (OUT_SVG / f'{safe}.svg').write_bytes(svg_bytes)
    (OUT_PNG / f'{safe}.png').write_bytes(png_bytes)

    n_obj  = len(entry.get('objects', []))
    n_attr = len(entry.get('attributes', []))
    n_rel  = len(entry.get('relationships', []))
    vc_str = f'  VC={norm_vc:.2f}' if norm_vc is not None else ''
    print(f'  {image_name}  [{vis_type}{vc_str}]  obj={n_obj}  attr={n_attr}  rel={n_rel}')
    print(f'    → {(OUT_PNG / f"{safe}.png").relative_to(ROOT)}')


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Render scene graphs — pure-genome condition')
    ap.add_argument('--input', type=str, default=None,
                    help=f'Input JSON (default: {DEFAULT_IN.name})')
    args = ap.parse_args()

    in_path = Path(args.input) if args.input else DEFAULT_IN
    if not in_path.exists():
        print(f'ERROR: input not found: {in_path}', file=sys.stderr)
        sys.exit(1)

    OUT_SVG.mkdir(parents=True, exist_ok=True)
    OUT_PNG.mkdir(parents=True, exist_ok=True)

    data = json.loads(in_path.read_text(encoding='utf-8'))
    print(f'Loaded {len(data)} images from {in_path.name}')

    # Load metadata
    meta: dict[str, dict] = {}
    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
        for _, row in df.iterrows():
            meta[row['imageName']] = {
                'VisType':      str(row.get('VisType', '')),
                'NormalizedVC': float(row.get('NormalizedVC', 0)),
            }

    print(f'\nRendering {len(data)} images...')
    print(f'  SVG → {OUT_SVG.relative_to(ROOT)}')
    print(f'  PNG → {OUT_PNG.relative_to(ROOT)}\n')

    rendered = 0
    skipped  = 0
    for image_name, entry in data.items():
        safe = image_name.removesuffix('.png').removesuffix('.jpg')
        if (OUT_PNG / f'{safe}.png').exists():
            skipped += 1
            continue
        m        = meta.get(image_name, {})
        vis_type = m.get('VisType', '')
        norm_vc  = m.get('NormalizedVC', None)
        render_image(image_name, entry, vis_type=vis_type, norm_vc=norm_vc)
        rendered += 1

    print(f'\nDone — {rendered} rendered, {skipped} skipped (already existed)'
          f'  →  {OUT_PNG.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
