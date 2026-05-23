"""Quick top-3 metrics computation for V0+TW-dyn (opus) and V0+T (for new-doc table)."""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from topic_select_preview import build_human_gt, compute_metrics, ROOT, ANCHORS


def load(csv_path):
    df = pd.read_csv(csv_path)
    df = df[~df['filename'].isin(ANCHORS)].copy()
    df['top3_topics'] = df['top3_topics'].fillna('')
    return df.rename(columns={'top3_topics': 'top3'})[['filename', 'top3']]


gt = build_human_gt()
runs = {
    'V0+T (opus)':           ROOT / 'results/vc_api_topicsel_v0_t/vc_scores.csv',
    'V0+TW=dyn (opus)':      ROOT / 'results/vc_api_topicsel_v0_tw_dyn_opus/vc_scores.csv',
    'V0+T+VT (opus)':        ROOT / 'results/vc_api_topicsel_v0_t_vt/vc_scores.csv',
    'V0+TW+VT [fixed-W]':    ROOT / 'results/vc_api_topicsel_v0_tw_vt/vc_scores.csv',
    'V0+TW=dyn+VT (opus)':   ROOT / 'results/vc_api_topicsel_v0_tw_dyn_vt/vc_scores.csv',
}

for label, p in runs.items():
    df = load(p)
    m = compute_metrics(df, gt)
    print(f'\n=== {label} ===')
    print(f'  F1_macro={m["macro_f1_mean_per_image"]:.4f}  IoU={m["macro_iou_mean_per_image"]:.4f}  '
          f'P_macro={m["macro_precision_mean"]:.4f}  R_macro={m["macro_recall_mean"]:.4f}  '
          f'microF1={m["micro_f1"]:.4f}  n={m["n_images"]}')
    print('  Per-topic F1:')
    for d, v in m['per_topic'].items():
        print(f'    {d:<18}: F1={v["f1"]:.3f}  P={v["precision"]:.3f}  R={v["recall"]:.3f}')
