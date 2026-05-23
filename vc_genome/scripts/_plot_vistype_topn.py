"""
Generate per-VisType top-N entity frequency bar charts (B vs V1).
Outputs three 3×3 grid PNGs to vc_genome_output_full/match/:
  vistype_grid_top_obj.png
  vistype_grid_top_attr.png
  vistype_grid_top_pred.png
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

ROOT      = Path(__file__).parent.parent
THREE_DIR = ROOT / 'vc_genome_output_full' / 'three_conditions'
OUT       = ROOT / 'vc_genome_output_full' / 'match'
PHRASES   = ROOT / 'phrase_reduction_v2' / 'image_compiled_phrases.csv'

sys.path.insert(0, str(Path(__file__).parent))
from _vc_canon import resolve_object_synset, resolve_predicate

NINE_VIS = ['Area', 'Bar', 'Cont.-ColorPatn', 'Glyph', 'Grid',
            'Line', 'Node-link', 'Point', 'Text']

COLORS = {'b': '#4C78A8', 'v1': '#F58518'}
LABELS = {'b': 'B (human-phrase)', 'v1': 'V1 (pure vision)'}
ALPHA  = 0.70
BIN_KW = dict(edgecolor='white', linewidth=0.3)


def build_counters(raw_data, images):
    obj_ctr  = Counter()
    attr_ctr = Counter()
    pred_ctr = Counter()
    for img in images:
        if img not in raw_data:
            continue
        ext = raw_data[img]
        for obj in ext['objects']:
            obj_ctr[resolve_object_synset(obj['name'])] += 1
        for attr in ext['attributes']:
            attr_ctr[attr.get('attr', attr.get('name', 'unknown'))] += 1
        for rel in ext['relationships']:
            pred_ctr[resolve_predicate(rel['pred'])] += 1
    return obj_ctr, attr_ctr, pred_ctr


def topN_barh_vt(ax, b_ctr, v1_ctr, vt, n_images, max_n=15):
    combined = b_ctr + v1_ctr
    n_avail  = len(combined)
    n        = min(max_n, n_avail) if n_avail > 0 else 0
    if n == 0:
        ax.set_title(f'{vt}\n(no data)', fontsize=8)
        ax.axis('off')
        return
    top_terms = [t for t, _ in combined.most_common(n)]
    y  = np.arange(len(top_terms))
    h  = 0.38
    ax.barh(y + h/2, [b_ctr[t]  for t in top_terms], height=h,
            color=COLORS['b'],  alpha=ALPHA, label=LABELS['b'],  **BIN_KW)
    ax.barh(y - h/2, [v1_ctr[t] for t in top_terms], height=h,
            color=COLORS['v1'], alpha=ALPHA, label=LABELS['v1'], **BIN_KW)
    ax.set_yticks(y)
    ax.set_yticklabels([t.replace('_', ' ') for t in top_terms], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel('Count', fontsize=7)
    ax.set_title(f'{vt}  (n={n_images})', fontsize=8.5, fontweight='bold')
    ax.legend(fontsize=6, loc='lower right')
    ax.tick_params(axis='x', labelsize=7)


def main():
    import pandas as pd

    print('Loading data...')
    with open(THREE_DIR / 'oar_B_510.json',  encoding='utf-8') as f:
        b_raw = json.load(f)
    with open(THREE_DIR / 'oar_V1_510.json', encoding='utf-8') as f:
        v1_raw = json.load(f)

    common = sorted(set(b_raw) & set(v1_raw))
    print(f'Common images: {len(common)}')

    phrases = pd.read_csv(PHRASES, usecols=['imageName', 'VisType']).drop_duplicates('imageName')
    vistype_map = phrases.set_index('imageName')['VisType'].to_dict()

    common_set = set(common)
    vt_images = {
        vt: [img for img, v in vistype_map.items() if v == vt and img in common_set]
        for vt in NINE_VIS
    }

    # Build counters per VisType
    vt_b  = {}
    vt_v1 = {}
    for vt in NINE_VIS:
        imgs = vt_images[vt]
        vt_b[vt]  = build_counters(b_raw,  imgs)
        vt_v1[vt] = build_counters(v1_raw, imgs)
        print(f'  {vt:20s}  n={len(imgs):3d}  '
              f'B_obj={sum(vt_b[vt][0].values()):4d}  '
              f'V1_obj={sum(vt_v1[vt][0].values()):5d}')

    entity_cfg = [
        (0, 'Object synsets',         'vistype_grid_top_obj.png'),
        (1, 'Attribute terms',         'vistype_grid_top_attr.png'),
        (2, 'Relationship predicates', 'vistype_grid_top_pred.png'),
    ]

    for ctr_idx, entity_label, fname in entity_cfg:
        print(f'\nPlotting {entity_label}...')
        fig, axes = plt.subplots(3, 3, figsize=(16, 16))
        for ax, vt in zip(axes.flatten(), NINE_VIS):
            topN_barh_vt(ax,
                         vt_b[vt][ctr_idx],
                         vt_v1[vt][ctr_idx],
                         vt,
                         n_images=len(vt_images[vt]),
                         max_n=15)
        plt.suptitle(
            f'Top-15 {entity_label} per VisType — B (human-phrase) vs V1 (pure vision)\n'
            f'(510-image full corpus)',
            fontsize=12, fontweight='bold', y=1.01
        )
        plt.tight_layout()
        save = OUT / fname
        fig.savefig(save, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved → {save}')

    print('\nDone.')


if __name__ == '__main__':
    main()
