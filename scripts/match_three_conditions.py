"""
Three-condition O/A/R pairwise comparison on the 43 eval images.

Reads:
    vc_genome_output_full/three_conditions/oar_B.json   (baseline: topics+phrases)
    vc_genome_output_full/three_conditions/oar_V1.json  (topics+image)
    vc_genome_output_full/three_conditions/oar_V2.json  (topics+image+anchors)

B is compared against V1 and V2 on the 43 non-anchor images.
V1 vs V2 is also reported.

Reports STRICT keys (same as match_api_vs_vision.py), RELAXED keys
(attribute-sign, predicate-family), and ROLE-LEVEL keys (Option A) that
project each canonical synset to a coarse chart-part role so instance-level
vision objects (e.g. `horizontal_bars` → mark.bar → data_area) align with
abstract human-phrase objects (e.g. `data_area` → structure.region →
data_area).

Outputs (vc_genome_output_full/three_conditions/match/):
    per_image_BvV1.csv, per_image_BvV2.csv, per_image_V1vV2.csv
    pair_summary.csv           — aggregate means per pair × metric
    unmapped_terms.csv
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from _vc_canon import canonicalize, normalize_attr, resolve_predicate  # noqa

ANCHOR_NAMES = {'VisC.503.6.png', 'InfoVisJ.619.17.png', 'InfoVisJ.1149.6(1).png'}

IN_DIR  = ROOT / 'vc_genome_output_full' / 'three_conditions'
OUT_DIR = IN_DIR / 'match'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Cross-vocabulary role projection (Option A) ────────────────────────
# Collapses canonical synsets to ~8 chart-part roles so that instance-level
# vision objects (e.g. `horizontal_bars` → mark.bar) align with abstract
# human-phrase objects (e.g. `data_area` → structure.region).
_ROLE_PREFIX = {
    'mark':      'data_area',      # bars, points, lines, areas, glyphs, nodes
    'structure': 'data_area',      # region, panel, grid, cell, layout, pattern, link
    'property':  'encoding',       # color, encoding, font_size
    'content':   'data_area',      # numbers, data, detail
    'whole':     'chart',          # visualization, map, graph
    'furniture': None,             # split below (legend vs axes)
    'text':      None,             # split below (title vs axis vs label)
    'unknown':   None,             # keep as-is
}
_ROLE_OVERRIDE = {
    'structure.background': 'background',
    'furniture.legend':     'legend',
    'furniture.axes':       'axes',
    'text.title':           'title',
    'text.axis_label':      'axes',
    'text.label':           'labels',
    'text.annotation':      'labels',
    'text.description':     'labels',
}


def to_role(synset: str) -> str:
    if synset in _ROLE_OVERRIDE:
        return _ROLE_OVERRIDE[synset]
    prefix = synset.split('.', 1)[0]
    role = _ROLE_PREFIX.get(prefix)
    if role is not None:
        return role
    # unknown.* — keep the tail so it still has a chance to match by name
    return synset


# ─── Predicate family rollup ─────────────────────────────────────────────
def predicate_family(pred: str) -> str:
    p = (pred or '').lower()
    rules = [
        ('aids',              ['aids', 'supports', 'facilitates', 'reinforces', 'helps', 'clarif', 'contextualiz']),
        ('simplifies',        ['simplif', 'reduces_complex', 'reduces_difficulty']),
        ('obscures',          ['obscur', 'clutter', 'overwhelm', 'distract', 'conflat']),
        ('hinders',           ['hinder', 'confuse', 'reduce_read', 'reduces_read', 'reduces_distinguish', 'fails']),
        ('increases_complex', ['increase', 'adds', 'complicat', 'amplif']),
        ('increases_effort',  ['effort', 'load', 'requires_', 'expertise']),
        ('describes',         ['describe', 'label', 'annotat', 'encode', 'represent', 'indicate', 'identif', 'show']),
        ('structures',        ['contain', 'organiz', 'surround', 'connect', 'group', 'overlap', 'adjacent']),
        ('co_occurs',         ['co_', 'combin', 'mix']),
        ('differentiates',    ['differen', 'contrast', 'distinguish']),
        ('compared',          ['compar', 'resembl']),
    ]
    for fam, keys in rules:
        if any(k in p for k in keys):
            return fam
    return 'other'


# ─── Key extractors ──────────────────────────────────────────────────────
def obj_keys(canon):
    return {o['synset'] for o in canon['objects']}


def obj_keys_role(canon):
    return {to_role(o['synset']) for o in canon['objects']}


def attr_keys_strict(canon):
    return {(a['object_synset'], a['attr']) for a in canon['attributes']}


def attr_keys_sign(canon):
    return {(a['object_synset'], a['sentiment']) for a in canon['attributes']}


def attr_keys_role_sign(canon):
    return {(to_role(a['object_synset']), a['sentiment']) for a in canon['attributes']}


def rel_triples(canon):
    return {(r['subject_synset'], r['predicate'], r['object_synset'])
            for r in canon['relationships']}


def rel_triples_family(canon):
    return {(r['subject_synset'], predicate_family(r['predicate']), r['object_synset'])
            for r in canon['relationships']}


def rel_triples_role_family(canon):
    return {(to_role(r['subject_synset']),
             predicate_family(r['predicate']),
             to_role(r['object_synset']))
            for r in canon['relationships']}


def rel_pairs(canon):
    return {frozenset((r['subject_synset'], r['object_synset']))
            for r in canon['relationships']}


def rel_pairs_role(canon):
    return {frozenset((to_role(r['subject_synset']), to_role(r['object_synset'])))
            for r in canon['relationships']}


def set_metrics(a: set, b: set):
    inter = a & b; union = a | b
    tp = len(inter); na, nb = len(a), len(b)
    jaccard = tp / len(union) if union else 0.0
    p = tp / nb if nb else 0.0
    r = tp / na if na else 0.0
    f1 = 2*p*r/(p+r) if (p+r) else 0.0
    return {
        'n_ref': na, 'n_pred': nb, 'inter': tp, 'union': len(union),
        'jaccard': round(jaccard, 4), 'precision': round(p, 4),
        'recall': round(r, 4), 'f1': round(f1, 4),
    }


# ─── Per-image comparison for one pair ───────────────────────────────────
LAYERS = [
    ('obj',             obj_keys),
    ('obj_role',        obj_keys_role),
    ('attr_strict',     attr_keys_strict),
    ('attr_sign',       attr_keys_sign),
    ('attr_role_sign',  attr_keys_role_sign),
    ('rel_strict',      rel_triples),
    ('rel_family',      rel_triples_family),
    ('rel_role_family', rel_triples_role_family),
    ('rel_loose',       rel_pairs),
    ('rel_role_loose',  rel_pairs_role),
]


def compare_pair(ref_oar, pred_oar, images):
    rows = []
    for img in images:
        ref = canonicalize(ref_oar[img])
        pred = canonicalize(pred_oar[img])
        row = {'imageName': img}
        for name, fn in LAYERS:
            m = set_metrics(fn(ref), fn(pred))
            for k, v in m.items():
                row[f'{name}_{k}'] = v
        rows.append(row)
    return pd.DataFrame(rows)


# ─── Main ────────────────────────────────────────────────────────────────
def main():
    oar_B  = json.loads((IN_DIR / 'oar_B.json').read_text(encoding='utf-8'))
    oar_V1 = json.loads((IN_DIR / 'oar_V1.json').read_text(encoding='utf-8'))
    oar_V2 = json.loads((IN_DIR / 'oar_V2.json').read_text(encoding='utf-8'))

    eval_imgs = sorted((set(oar_B) | set(oar_V1) | set(oar_V2)) - ANCHOR_NAMES)
    # Only keep images present in all three
    eval_imgs = [i for i in eval_imgs if i in oar_B and i in oar_V1 and i in oar_V2]
    print(f'Evaluation images present in all 3 conditions: {len(eval_imgs)}')

    pairs = [
        ('B_vs_V1', oar_B,  oar_V1),
        ('B_vs_V2', oar_B,  oar_V2),
        ('V1_vs_V2', oar_V1, oar_V2),
    ]

    summaries = []
    for tag, ref, pred in pairs:
        df = compare_pair(ref, pred, eval_imgs)
        df.to_csv(OUT_DIR / f'per_image_{tag}.csv', index=False)
        print(f'Wrote per_image_{tag}.csv ({len(df)} rows)')
        means = df.drop(columns=['imageName']).mean(numeric_only=True).round(4)
        means.name = tag
        summaries.append(means)

    summary = pd.concat(summaries, axis=1)
    summary.index.name = 'metric'
    summary.to_csv(OUT_DIR / 'pair_summary.csv')

    # Focused headline table
    headline_rows = []
    layer_names = ['obj', 'obj_role',
                   'attr_strict', 'attr_sign', 'attr_role_sign',
                   'rel_strict', 'rel_family', 'rel_role_family',
                   'rel_loose', 'rel_role_loose']
    for tag in ['B_vs_V1', 'B_vs_V2', 'V1_vs_V2']:
        r = {'pair': tag}
        for layer in layer_names:
            r[f'{layer}_f1']      = summary.loc[f'{layer}_f1', tag]
            r[f'{layer}_jaccard'] = summary.loc[f'{layer}_jaccard', tag]
        headline_rows.append(r)
    headline = pd.DataFrame(headline_rows)
    headline.to_csv(OUT_DIR / 'headline.csv', index=False)

    print('\n── Mean F1 (reference ↔ prediction) on 43 eval images ──')
    print('Strict (synset):')
    print(headline[['pair', 'obj_f1', 'attr_strict_f1', 'attr_sign_f1',
                    'rel_strict_f1', 'rel_family_f1', 'rel_loose_f1']]
          .to_string(index=False))
    print('\nRole-level (Option A: synsets → chart-part roles):')
    print(headline[['pair', 'obj_role_f1', 'attr_role_sign_f1',
                    'rel_role_family_f1', 'rel_role_loose_f1']]
          .to_string(index=False))

    print('\n── Mean Jaccard ──')
    print('Strict (synset):')
    print(headline[['pair', 'obj_jaccard', 'attr_strict_jaccard', 'attr_sign_jaccard',
                    'rel_strict_jaccard', 'rel_family_jaccard', 'rel_loose_jaccard']]
          .to_string(index=False))
    print('\nRole-level:')
    print(headline[['pair', 'obj_role_jaccard', 'attr_role_sign_jaccard',
                    'rel_role_family_jaccard', 'rel_role_loose_jaccard']]
          .to_string(index=False))


if __name__ == '__main__':
    main()
