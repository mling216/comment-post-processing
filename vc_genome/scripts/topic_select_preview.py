"""
Topic Selection Analysis — Preview (V1 only)
=============================================
Builds human ground-truth topic sets from image_compiled_phrases.csv for the
43 non-anchor evaluation images, derives V1 top-3 topics from existing
per-dimension VC scores (highest 3 dims), and reports F1 / IoU / per-topic
precision-recall.

Outputs (written to `topic_selection/`):
  - human_topic_gt.csv         : filename, human_topics (;-joined dim keys)
  - v1_top3_predictions.csv    : filename, top3 (;-joined), source run
  - v1_topic_metrics.json      : aggregated metrics
"""

import json
from pathlib import Path
import pandas as pd

ROOT = Path(r'd:\Coding\Copilot\comment_post_processing')
PHRASE_DIR = ROOT / 'phrase_reduction_v2'
CLAUDE_DIR = ROOT / 'Claude_vc_prediction'
OUT_DIR = ROOT / 'topic_selection'
OUT_DIR.mkdir(exist_ok=True)

ANCHORS = {'VisC.503.6.png', 'InfoVisJ.619.17.png', 'InfoVisJ.1149.6(1).png'}

# Human-topic-label -> VC-dim-key (1:1 mapping from subagent report)
TOPIC_TO_DIM = {
    'Data Density / Image Clutter': 'data_density',
    'Visual Encoding Clarity': 'visual_encoding',
    'Semantics / Text Legibility': 'text_annotation',
    'Schema': 'domain_schema',
    'Color, Symbol, and Texture Details': 'color_symbol',
    'Aesthetics Uncertainty': 'aesthetic_order',
    'Immediacy / Cognitive Load': 'cognitive_load',
}
DIMS = list(TOPIC_TO_DIM.values())


# ── Build human GT ──────────────────────────────────────────────────────────
def build_human_gt() -> pd.DataFrame:
    gt46 = pd.read_csv(CLAUDE_DIR / 'gt_all_46.csv')
    compiled = pd.read_csv(PHRASE_DIR / 'image_compiled_phrases.csv')

    eval_imgs = [n for n in gt46['imageName'].tolist() if n not in ANCHORS]
    assert len(eval_imgs) == 43, f'expected 43 eval imgs, got {len(eval_imgs)}'

    rows = []
    missing = []
    for img in eval_imgs:
        m = compiled[compiled['imageName'] == img]
        if len(m) == 0:
            missing.append(img)
            continue
        topics_raw = str(m.iloc[0]['Topics'])
        topic_labels = [t.strip() for t in topics_raw.split(';') if t.strip()]
        dims = []
        unmapped = []
        for t in topic_labels:
            if t in TOPIC_TO_DIM:
                d = TOPIC_TO_DIM[t]
                if d not in dims:
                    dims.append(d)
            else:
                unmapped.append(t)
        if unmapped:
            print(f'  WARN {img}: unmapped topics {unmapped}')
        rows.append({
            'filename': img,
            'n_human_topics': len(dims),
            'human_topics': ';'.join(dims),
        })

    if missing:
        print(f'MISSING from image_compiled_phrases.csv: {missing}')
    df = pd.DataFrame(rows)
    return df


# ── Derive V1 top-3 ─────────────────────────────────────────────────────────
def derive_v1_top3(vc_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(vc_csv)
    out_rows = []
    for _, r in df.iterrows():
        if r['filename'] in ANCHORS:
            continue
        scores = {d: float(r[d]) for d in DIMS}
        # sort desc; break ties by DIMS order (stable)
        ranked = sorted(DIMS, key=lambda d: -scores[d])
        top3 = ranked[:3]
        out_rows.append({'filename': r['filename'], 'top3': ';'.join(top3),
                         **{f'score_{d}': scores[d] for d in DIMS}})
    return pd.DataFrame(out_rows)


def derive_v1_mean_top3(r1_csv: Path, r2_csv: Path) -> pd.DataFrame:
    r1 = pd.read_csv(r1_csv).set_index('filename')
    r2 = pd.read_csv(r2_csv).set_index('filename')
    common = r1.index.intersection(r2.index)
    rows = []
    for f in common:
        if f in ANCHORS:
            continue
        scores = {d: (float(r1.loc[f, d]) + float(r2.loc[f, d])) / 2 for d in DIMS}
        ranked = sorted(DIMS, key=lambda d: -scores[d])
        rows.append({'filename': f, 'top3': ';'.join(ranked[:3]),
                     **{f'score_{d}': scores[d] for d in DIMS}})
    return pd.DataFrame(rows)


# ── Metrics ────────────────────────────────────────────────────────────────
def f1_set(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def iou_set(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    u = pred | gold
    return len(pred & gold) / len(u) if u else 0.0


def compute_metrics(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict:
    merged = pred_df.merge(gt_df, on='filename', how='inner')
    per_image = []
    # for per-topic aggregates
    topic_tp = {d: 0 for d in DIMS}
    topic_fp = {d: 0 for d in DIMS}
    topic_fn = {d: 0 for d in DIMS}
    # micro counters
    micro_tp = micro_fp = micro_fn = 0

    for _, r in merged.iterrows():
        pred = set(r['top3'].split(';')) if r['top3'] else set()
        gold = set(r['human_topics'].split(';')) if r['human_topics'] else set()
        per_image.append({
            'filename': r['filename'],
            'n_gold': len(gold),
            'n_pred': len(pred),
            'n_intersect': len(pred & gold),
            'f1': f1_set(pred, gold),
            'iou': iou_set(pred, gold),
            'precision': len(pred & gold) / len(pred) if pred else 0.0,
            'recall': len(pred & gold) / len(gold) if gold else 0.0,
        })
        for d in DIMS:
            if d in pred and d in gold:
                topic_tp[d] += 1; micro_tp += 1
            elif d in pred and d not in gold:
                topic_fp[d] += 1; micro_fp += 1
            elif d not in pred and d in gold:
                topic_fn[d] += 1; micro_fn += 1

    pi_df = pd.DataFrame(per_image)

    per_topic = {}
    for d in DIMS:
        tp, fp, fn = topic_tp[d], topic_fp[d], topic_fn[d]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2*p*r/(p+r) if (p+r) else 0.0
        per_topic[d] = {'tp': tp, 'fp': fp, 'fn': fn,
                         'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f1, 4)}

    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r) if (micro_p+micro_r) else 0.0

    return {
        'n_images': len(pi_df),
        'macro_f1_mean_per_image': round(pi_df['f1'].mean(), 4),
        'macro_iou_mean_per_image': round(pi_df['iou'].mean(), 4),
        'macro_precision_mean': round(pi_df['precision'].mean(), 4),
        'macro_recall_mean': round(pi_df['recall'].mean(), 4),
        'micro_precision': round(micro_p, 4),
        'micro_recall': round(micro_r, 4),
        'micro_f1': round(micro_f1, 4),
        'per_topic': per_topic,
        'per_image_df': pi_df,
    }


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Building human GT ...')
    gt = build_human_gt()
    gt.to_csv(OUT_DIR / 'human_topic_gt.csv', index=False)
    print(f'  Wrote human_topic_gt.csv ({len(gt)} imgs)')
    print(f'  # humans per image distribution:\n{gt["n_human_topics"].value_counts().sort_index()}\n')

    runs = {
        'V1_r1': ROOT / 'results' / 'vc_api_46gt_v3' / 'vc_scores.csv',
        'V1_r2': ROOT / 'results' / 'vc_api_46gt_v1_r2' / 'vc_scores.csv',
    }

    results = {}
    for label, csv in runs.items():
        pred = derive_v1_top3(csv)
        pred.to_csv(OUT_DIR / f'{label}_top3.csv', index=False)
        m = compute_metrics(pred, gt)
        m_export = {k: v for k, v in m.items() if k != 'per_image_df'}
        results[label] = m_export
        print(f'\n=== {label} ===')
        print(f'  n={m["n_images"]}  F1 (macro-per-image)={m["macro_f1_mean_per_image"]}  IoU={m["macro_iou_mean_per_image"]}  micro-F1={m["micro_f1"]}')
        print(f'  macro Prec={m["macro_precision_mean"]}  Recall={m["macro_recall_mean"]}')
        print('  per-topic P/R/F1:')
        for d, v in m['per_topic'].items():
            print(f'    {d:18s}  P={v["precision"]:.3f}  R={v["recall"]:.3f}  F1={v["f1"]:.3f}  (tp={v["tp"]}, fp={v["fp"]}, fn={v["fn"]})')

    # Mean of V1 r1+r2 score averages
    pred_mean = derive_v1_mean_top3(runs['V1_r1'], runs['V1_r2'])
    pred_mean.to_csv(OUT_DIR / 'V1_mean_top3.csv', index=False)
    m = compute_metrics(pred_mean, gt)
    m_export = {k: v for k, v in m.items() if k != 'per_image_df'}
    results['V1_mean'] = m_export
    print(f'\n=== V1 (mean of r1+r2 scores, then top-3) ===')
    print(f'  n={m["n_images"]}  F1={m["macro_f1_mean_per_image"]}  IoU={m["macro_iou_mean_per_image"]}  micro-F1={m["micro_f1"]}')
    print('  per-topic P/R/F1:')
    for d, v in m['per_topic'].items():
        print(f'    {d:18s}  P={v["precision"]:.3f}  R={v["recall"]:.3f}  F1={v["f1"]:.3f}  (tp={v["tp"]}, fp={v["fp"]}, fn={v["fn"]})')

    with open(OUT_DIR / 'v1_topic_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nWrote v1_topic_metrics.json')
