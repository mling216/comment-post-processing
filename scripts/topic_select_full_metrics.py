"""
Topic Selection Analysis — Full Metrics Report
===============================================
Computes F1 / IoU / per-topic P/R for all variants:
  V0+T (opus-4.6, t=0, prompt top3)
  V0+TW (opus-4.6, t=0, prompt top3)
  V1 (opus-4.6, t=1 r1+r2 mean, derived top3 from per-dim scores)
  V1 (opus-4.6, t=0, derived top3 from per-dim scores)

Outputs:
  topic_selection/topic_metrics_full.json
  topic_selection/topic_summary.csv
  topic_selection/per_topic_f1.csv
  topic_selection/per_image_all.csv
"""

import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from topic_select_preview import (
    DIMS, ANCHORS, build_human_gt, derive_v1_top3, derive_v1_mean_top3,
    compute_metrics, OUT_DIR, ROOT,
)


def load_v0_top3(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[~df['filename'].isin(ANCHORS)].copy()
    df['top3_topics'] = df['top3_topics'].fillna('')
    return df.rename(columns={'top3_topics': 'top3'})[['filename', 'top3']]


if __name__ == '__main__':
    gt = build_human_gt()
    gt.to_csv(OUT_DIR / 'human_topic_gt.csv', index=False)

    runs = {}

    # V0+T (prompted top-3)
    p = ROOT / 'results' / 'vc_api_topicsel_v0_t' / 'vc_scores.csv'
    runs['V0+T'] = load_v0_top3(p)

    # V0+TW (prompted top-3)
    p = ROOT / 'results' / 'vc_api_topicsel_v0_tw' / 'vc_scores.csv'
    runs['V0+TW'] = load_v0_top3(p)

    # V1 runs (derived from per-dim scores)
    runs['V1 (t=1, r1)'] = derive_v1_top3(ROOT / 'results' / 'vc_api_46gt_v3' / 'vc_scores.csv')
    runs['V1 (t=1, r2)'] = derive_v1_top3(ROOT / 'results' / 'vc_api_46gt_v1_r2' / 'vc_scores.csv')
    runs['V1 (t=1, mean)'] = derive_v1_mean_top3(
        ROOT / 'results' / 'vc_api_46gt_v3' / 'vc_scores.csv',
        ROOT / 'results' / 'vc_api_46gt_v1_r2' / 'vc_scores.csv',
    )
    runs['V1 (t=0)'] = derive_v1_top3(ROOT / 'results' / 'vc_api_46gt_v1_t0' / 'vc_scores.csv')

    summary_rows = []
    per_topic_rows = []
    per_image_frames = []
    full = {}

    for label, pred_df in runs.items():
        m = compute_metrics(pred_df[['filename', 'top3']], gt)
        full[label] = {k: v for k, v in m.items() if k != 'per_image_df'}
        summary_rows.append({
            'variant': label,
            'n': m['n_images'],
            'F1_macro_image': m['macro_f1_mean_per_image'],
            'IoU_macro_image': m['macro_iou_mean_per_image'],
            'Precision_macro': m['macro_precision_mean'],
            'Recall_macro': m['macro_recall_mean'],
            'F1_micro': m['micro_f1'],
            'Precision_micro': m['micro_precision'],
            'Recall_micro': m['micro_recall'],
        })
        for d, v in m['per_topic'].items():
            per_topic_rows.append({'variant': label, 'topic': d, **v})

        # save per-image
        pi = m['per_image_df'].copy()
        pi['variant'] = label
        per_image_frames.append(pi)

        print(f'{label:18s}  F1={m["macro_f1_mean_per_image"]:.4f}  IoU={m["macro_iou_mean_per_image"]:.4f}  micro-F1={m["micro_f1"]:.4f}  (n={m["n_images"]})')

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / 'topic_summary.csv', index=False)

    per_topic = pd.DataFrame(per_topic_rows)
    per_topic.to_csv(OUT_DIR / 'per_topic_f1.csv', index=False)

    pd.concat(per_image_frames).to_csv(OUT_DIR / 'per_image_all.csv', index=False)

    with open(OUT_DIR / 'topic_metrics_full.json', 'w') as f:
        json.dump(full, f, indent=2)

    # Print wide per-topic F1 table
    print('\n=== Per-topic F1 (rows=variant, cols=topic) ===')
    pv = per_topic.pivot(index='variant', columns='topic', values='f1')
    # reorder columns to canonical DIMS
    pv = pv[DIMS]
    print(pv.round(3).to_string())

    print('\n=== Summary ===')
    print(summary.to_string(index=False))
