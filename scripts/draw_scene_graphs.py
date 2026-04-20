"""
Draw scene graph PNGs for ablation extractions.
Uses graphviz to render each image's scene graph.

Usage:
    python scripts/draw_scene_graphs.py          # default: v3
    python scripts/draw_scene_graphs.py v4       # render v4
    python scripts/draw_scene_graphs.py v3 v4    # render both
"""
import json, os, sys, argparse
from pathlib import Path
from collections import defaultdict

import graphviz

# Graphviz setup for Windows
os.environ['PATH'] = r'C:\Program Files (x86)\Graphviz\bin;' + os.environ.get('PATH', '')

# ── Colors ──
COLOR_OBJECT   = '#b8e6b8'
COLOR_ATTR_POS = '#c7b8e6'
COLOR_ATTR_NEG = '#e8dff5'
COLOR_RELATION = '#f5b8b8'


def build_scene_graph_dot(extraction, image_name):
    """Build a graphviz Digraph from raw v3 extraction."""
    dot = graphviz.Digraph(
        name=image_name, format='png',
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

    # Build id→name map
    id_to_name = {}
    for obj in extraction['objects']:
        oid = str(obj['id'])
        name = obj['name'].replace('_', ' ')
        id_to_name[oid] = name
        dot.node(f'obj_{oid}', label=name,
                 fillcolor=COLOR_OBJECT, color='#2c3e50',
                 fontcolor='#1a1a1a', penwidth='1.8')

    # Attributes
    for i, attr in enumerate(extraction['attributes']):
        oid = str(attr['object_id'])
        # Handle inconsistent sentiment key formats
        sentiment = attr.get('sentiment', attr.get('+', attr.get('-', '?')))
        if sentiment == '?':
            # Fallback: check if '+' or '-' appears as a key
            sentiment = '-' if '-' in attr else ('+' if '+' in attr else '?')
        sent = '(+)' if sentiment == '+' else '(\u2212)'
        label = f"{sent} {attr['attr'].replace('_', ' ')[:35]}"
        color = COLOR_ATTR_POS if sentiment == '+' else COLOR_ATTR_NEG
        attr_id = f'attr_{i}'
        dot.node(attr_id, label=label, fillcolor=color,
                 color='#7f8c8d', fontcolor='#1a1a1a')
        dot.edge(f'obj_{oid}', attr_id, color='#aaaaaa', arrowsize='0.7')

    # Relationships
    for ri, rel in enumerate(extraction['relationships']):
        subj_id = str(rel['subj'])
        obj_id = str(rel['obj'])
        pred_label = rel['pred'].replace('_', ' ')
        pred_node = f'rel_{ri}'
        dot.node(pred_node, label=pred_label,
                 fillcolor=COLOR_RELATION, color='#d46a6a', fontcolor='#8b0000')
        dot.edge(f'obj_{subj_id}', pred_node,
                 color='#d46a6a', penwidth='1.5', arrowsize='0.8')
        dot.edge(pred_node, f'obj_{obj_id}',
                 color='#d46a6a', penwidth='1.5', arrowsize='0.8')

    # Title
    n_obj = len(extraction['objects'])
    n_attr = len(extraction['attributes'])
    n_rel = len(extraction['relationships'])
    title_html = f'''<
    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2" CELLPADDING="2">
    <TR><TD COLSPAN="8" ALIGN="CENTER"><B><FONT POINT-SIZE="14">{image_name}</FONT></B></TD></TR>
    <TR><TD COLSPAN="8" ALIGN="CENTER"><FONT POINT-SIZE="11">{n_obj} obj | {n_attr} attr | {n_rel} rel</FONT></TD></TR>
    <TR>
      <TD BGCOLOR="#b8e6b8" BORDER="1" STYLE="rounded" WIDTH="20"> </TD><TD>Object</TD>
      <TD BGCOLOR="#c7b8e6" BORDER="1" STYLE="rounded" WIDTH="20"> </TD><TD>Attr (+)</TD>
      <TD BGCOLOR="#e8dff5" BORDER="1" STYLE="rounded" WIDTH="20"> </TD><TD>Attr (\u2212)</TD>
      <TD BGCOLOR="#f5b8b8" BORDER="1" STYLE="rounded" WIDTH="20"> </TD><TD>Relationship</TD>
    </TR>
    </TABLE>>'''
    dot.attr(label=title_html, labelloc='t', labeljust='c')
    return dot


def main():
    parser = argparse.ArgumentParser(description='Draw scene graph PNGs for ablation extractions')
    parser.add_argument('versions', nargs='*', default=['v3'], help='Which version(s) to render (default: v3)')
    args = parser.parse_args()

    os.chdir(Path(__file__).parent.parent)

    for version in args.versions:
        src_file = Path(f'vc_genome_output_full/ablation/llm_extractions_{version}.json')
        out_dir = Path(f'vc_genome_output_full/ablation/scene_graphs_{version}')
        out_dir.mkdir(parents=True, exist_ok=True)

        if not src_file.exists():
            print(f'ERROR: {src_file} not found, skipping {version}')
            continue

        with open(src_file, 'r', encoding='utf-8') as f:
            extractions = json.load(f)

        print(f'\n{"="*50}')
        print(f'Version: {version.upper()} — {len(extractions)} extractions')
        print(f'Output:  {out_dir}/')

        # Suppress fontconfig stderr warnings
        _stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')

        try:
            for i, (img, ext) in enumerate(extractions.items(), 1):
                dot = build_scene_graph_dot(ext, img)
                safe_name = img.removesuffix('.png')
                png_data = dot.pipe(format='png')
                (out_dir / f'{safe_name}.png').write_bytes(png_data)
                print(f'  [{i}/{len(extractions)}] {img} ({len(png_data)//1024} KB)')
        finally:
            sys.stderr.close()
            sys.stderr = _stderr

        print(f'Done. Saved {len(extractions)} PNGs to {out_dir}/')


if __name__ == '__main__':
    main()
