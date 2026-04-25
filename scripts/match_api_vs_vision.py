"""
Human-vs-Vision O/A/R Match
============================
Compare the API-grounded extraction (human comments/phrases) against the
vision-grounded extraction (image pixels) at three layers: Objects,
Attributes, and Relationships.

Inputs:
    vc_genome_output_full/llm_extractions_api.json      (Stage 1, human-grounded)
    vc_genome_output_full/llm_extractions_vision.json   (Stage 2, vision-grounded)

Outputs (vc_genome_output_full/match/):
    per_image_metrics.csv   — one row per image, all layer metrics
    layer_summary.csv       — aggregate mean/median/sd per layer
    unmapped_terms.csv      — objects/predicates that fell through to `unknown.*`
    qualitative_slice.md    — side-by-side dump for high/median/low agreement images

Matching is purely set-based on canonical keys; VisType and NormalizedVC
are intentionally NOT used.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from _vc_canon import canonicalize  # noqa: E402

API_FILE    = ROOT / 'vc_genome_output_full' / 'llm_extractions_api.json'
VISION_FILE = ROOT / 'vc_genome_output_full' / 'llm_extractions_vision.json'
OUT_DIR     = ROOT / 'vc_genome_output_full' / 'match'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Key extractors ──────────────────────────────────────────────────────
def object_keys(canon):
    """Set of canonical object synsets (dropping trivially unknown ones keeps
    set size sane but we still report coverage elsewhere)."""
    return {o['synset'] for o in canon['objects']}


def attribute_keys(canon):
    """(object_synset, normalized_attr) pairs."""
    return {(a['object_synset'], a['attr']) for a in canon['attributes']}


def rel_triples_strict(canon):
    """(subj_synset, predicate, obj_synset)"""
    return {(r['subject_synset'], r['predicate'], r['object_synset'])
            for r in canon['relationships']}


def rel_pairs_loose(canon):
    """Unordered {subj_synset, obj_synset} pair, ignoring predicate."""
    return {frozenset((r['subject_synset'], r['object_synset']))
            for r in canon['relationships']}


# ─── Metric helpers ──────────────────────────────────────────────────────
def set_metrics(a: set, b: set) -> dict:
    """Treat `a` (api / human-grounded) as reference, `b` (vision) as prediction."""
    inter = a & b
    union = a | b
    tp = len(inter)
    n_a, n_b = len(a), len(b)
    jaccard   = tp / len(union) if union else 0.0
    precision = tp / n_b if n_b else 0.0   # vision correctness vs human
    recall    = tp / n_a if n_a else 0.0   # vision coverage of human
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        'n_api': n_a, 'n_vision': n_b, 'intersection': tp,
        'union': len(union),
        'jaccard': round(jaccard, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
    }


def sentiment_agreement(a_items, b_items, key_fn):
    """Among elements matched by key, what fraction share the same sentiment?"""
    a_map = {}
    for x in a_items:
        k = key_fn(x)
        # If duplicates, keep first — sentiment should be consistent within a side.
        a_map.setdefault(k, x.get('sentiment'))
    b_map = {}
    for x in b_items:
        k = key_fn(x)
        b_map.setdefault(k, x.get('sentiment'))
    shared = a_map.keys() & b_map.keys()
    if not shared:
        return None, 0
    agree = sum(1 for k in shared if a_map[k] == b_map[k])
    return round(agree / len(shared), 4), len(shared)


# ─── Per-image comparison ────────────────────────────────────────────────
def compare_one(image_name, api_raw, vis_raw):
    a = canonicalize(api_raw)
    v = canonicalize(vis_raw)

    obj_a, obj_v = object_keys(a), object_keys(v)
    attr_a, attr_v = attribute_keys(a), attribute_keys(v)
    rel_strict_a, rel_strict_v = rel_triples_strict(a), rel_triples_strict(v)
    rel_loose_a,  rel_loose_v  = rel_pairs_loose(a),    rel_pairs_loose(v)

    row = {'imageName': image_name}

    for label, sa, sb in [
        ('obj',       obj_a,        obj_v),
        ('attr',      attr_a,       attr_v),
        ('rel_strict', rel_strict_a, rel_strict_v),
        ('rel_loose',  rel_loose_a,  rel_loose_v),
    ]:
        m = set_metrics(sa, sb)
        for k, val in m.items():
            row[f'{label}_{k}'] = val

    # Sentiment agreement on matched subsets
    attr_sent, attr_shared = sentiment_agreement(
        a['attributes'], v['attributes'],
        key_fn=lambda x: (x['object_synset'], x['attr']),
    )
    rel_sent, rel_shared = sentiment_agreement(
        a['relationships'], v['relationships'],
        key_fn=lambda x: (x['subject_synset'], x['predicate'], x['object_synset']),
    )
    row['attr_sentiment_agreement'] = attr_sent
    row['attr_sentiment_n']         = attr_shared
    row['rel_sentiment_agreement']  = rel_sent
    row['rel_sentiment_n']          = rel_shared

    # Coverage: fraction of each side that canonicalized cleanly
    def _frac_mapped(syns):
        if not syns:
            return None
        return round(sum(1 for s in syns if not s.startswith('unknown.')) / len(syns), 4)

    row['obj_api_mapped_frac']    = _frac_mapped([o['synset'] for o in a['objects']])
    row['obj_vision_mapped_frac'] = _frac_mapped([o['synset'] for o in v['objects']])

    return row, a, v


# ─── Unmapped collection ─────────────────────────────────────────────────
def collect_unmapped(canon_list_by_side):
    """Return rows for objects/predicates that stayed `unknown.*` after canon."""
    rows = []
    for side, canon_list in canon_list_by_side.items():
        obj_ctr, pred_ctr = Counter(), Counter()
        for canon in canon_list:
            for o in canon['objects']:
                if o['synset'].startswith('unknown.'):
                    obj_ctr[o['name']] += 1
            for r in canon['relationships']:
                # predicates are passed through verbatim when unmapped;
                # heuristic: a predicate is "unmapped" if it contains an underscore
                # token not present in the canon vocab — we just report all non-canon.
                from _vc_canon import PREDICATE_CANON, resolve_predicate  # noqa
                if r['predicate'] not in PREDICATE_CANON.values():
                    pred_ctr[r['predicate']] += 1
        for name, c in obj_ctr.most_common():
            rows.append({'side': side, 'kind': 'object', 'term': name, 'count': c})
        for name, c in pred_ctr.most_common():
            rows.append({'side': side, 'kind': 'predicate', 'term': name, 'count': c})
    return rows


# ─── Qualitative slice ───────────────────────────────────────────────────
def _render_side_md(title, canon):
    lines = [f'**{title}**', '']
    lines.append(f'_Objects ({len(canon["objects"])})_: ' +
                 ', '.join(sorted({o['synset'] for o in canon['objects']})))
    lines.append('')
    lines.append(f'_Attributes ({len(canon["attributes"])})_:')
    for a in canon['attributes']:
        lines.append(f'- `{a["object_synset"]}` · {a["attr"]} [{a["sentiment"]}] '
                     f'({a["subtopic"]})')
    lines.append('')
    lines.append(f'_Relationships ({len(canon["relationships"])})_:')
    for r in canon['relationships']:
        lines.append(f'- `{r["subject_synset"]}` --{r["predicate"]}--> '
                     f'`{r["object_synset"]}` [{r["sentiment"]}]')
    lines.append('')
    return '\n'.join(lines)


def qualitative_slice(per_image_df, api_canon, vis_canon, k=1):
    """Pick the top-k highest, median, and lowest agreement images (by obj_f1)."""
    df = per_image_df.dropna(subset=['obj_f1']).sort_values('obj_f1')
    if df.empty:
        return ''
    picks = []
    n = len(df)
    picks += [(df.iloc[i]['imageName'], 'low')    for i in range(min(k, n))]
    mid = n // 2
    picks += [(df.iloc[mid]['imageName'], 'median')]
    picks += [(df.iloc[-(i + 1)]['imageName'], 'high') for i in range(min(k, n))]

    out = ['# Qualitative Slice: Human vs Vision O/A/R', '',
           'Images selected by object-F1 rank.', '']
    for img, tag in picks:
        row = df[df['imageName'] == img].iloc[0]
        out.append(f'## {tag.upper()} — `{img}` (obj_f1={row["obj_f1"]:.2f}, '
                   f'attr_f1={row["attr_f1"]:.2f}, rel_strict_f1={row["rel_strict_f1"]:.2f})')
        out.append('')
        out.append(_render_side_md('API (human-grounded)', api_canon[img]))
        out.append(_render_side_md('Vision (image-grounded)', vis_canon[img]))
        out.append('\n---\n')
    return '\n'.join(out)


# ─── Main ────────────────────────────────────────────────────────────────
def main():
    api    = json.loads(API_FILE.read_text(encoding='utf-8'))
    vision = json.loads(VISION_FILE.read_text(encoding='utf-8'))

    shared = sorted(set(api) & set(vision))
    api_only = sorted(set(api) - set(vision))
    vis_only = sorted(set(vision) - set(api))
    print(f'Images: api={len(api)} vision={len(vision)} shared={len(shared)} '
          f'api_only={len(api_only)} vision_only={len(vis_only)}')

    rows = []
    api_canon_map, vis_canon_map = {}, {}
    for img in shared:
        row, a_canon, v_canon = compare_one(img, api[img], vision[img])
        rows.append(row)
        api_canon_map[img] = a_canon
        vis_canon_map[img] = v_canon

    per_image = pd.DataFrame(rows)
    per_image.to_csv(OUT_DIR / 'per_image_metrics.csv', index=False)
    print(f'Wrote per_image_metrics.csv  ({len(per_image)} rows)')

    # Aggregate
    metric_cols = [c for c in per_image.columns
                   if c != 'imageName' and per_image[c].dtype != object]
    agg = per_image[metric_cols].agg(['mean', 'median', 'std']).T.round(4)
    agg.index.name = 'metric'
    agg.to_csv(OUT_DIR / 'layer_summary.csv')
    print(f'Wrote layer_summary.csv     ({len(agg)} metrics)')

    # Unmapped terms
    unmapped = collect_unmapped({
        'api':    list(api_canon_map.values()),
        'vision': list(vis_canon_map.values()),
    })
    pd.DataFrame(unmapped).to_csv(OUT_DIR / 'unmapped_terms.csv', index=False)
    print(f'Wrote unmapped_terms.csv    ({len(unmapped)} rows)')

    # Qualitative slice
    md = qualitative_slice(per_image, api_canon_map, vis_canon_map, k=1)
    (OUT_DIR / 'qualitative_slice.md').write_text(md, encoding='utf-8')
    print('Wrote qualitative_slice.md')

    # Console summary
    headline = ['obj_jaccard', 'obj_f1',
                'attr_jaccard', 'attr_f1',
                'rel_strict_jaccard', 'rel_strict_f1',
                'rel_loose_jaccard',  'rel_loose_f1',
                'attr_sentiment_agreement', 'rel_sentiment_agreement']
    print('\n── Mean per-image agreement (api ↔ vision) ──')
    for m in headline:
        if m in per_image.columns:
            print(f'  {m:28s} {per_image[m].mean():.3f}')


if __name__ == '__main__':
    main()
