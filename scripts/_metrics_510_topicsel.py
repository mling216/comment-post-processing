"""
Topic Selection Metrics — 510-image expansion (V0+T and V0+TW)
===============================================================
Computes macro-F1 overall and per-vis-type for the top-3 topic-selection
task on the full 510-image set. Compared against the 63-image pilot numbers.

Usage:
    python _metrics_510_topicsel.py
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent

# reuse compute_metrics / f1_set from topic_select_preview
sys.path.insert(0, str(Path(__file__).parent))
from topic_select_preview import DIMS, TOPIC_TO_DIM, compute_metrics

COMPILED = ROOT / 'phrase_reduction_v2' / 'image_compiled_phrases.csv'
INPUT_510 = ROOT / 'results' / 'vc_api_510_v0_tw_input.csv'


def build_gt_510() -> pd.DataFrame:
    """Build human topic GT for all 510 images."""
    compiled = pd.read_csv(COMPILED)[['imageName', 'Topics']]
    inp = pd.read_csv(INPUT_510)[['imageName', 'VisType']]
    merged = inp.merge(compiled, on='imageName')

    rows = []
    for _, r in merged.iterrows():
        topics_raw = str(r['Topics'])
        topic_labels = [t.strip() for t in topics_raw.split(';') if t.strip()]
        dims = []
        for t in topic_labels:
            if t in TOPIC_TO_DIM:
                d = TOPIC_TO_DIM[t]
                if d not in dims:
                    dims.append(d)
        rows.append({
            'filename': r['imageName'],
            'VisType':  r['VisType'],
            'n_human_topics': len(dims),
            'human_topics': ';'.join(dims),
        })
    return pd.DataFrame(rows)


def load_preds(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['top3_topics'] = df['top3_topics'].fillna('')
    return df.rename(columns={'filename': 'filename', 'top3_topics': 'top3'})[['filename', 'top3']]


def run(label: str, csv_path: Path, gt: pd.DataFrame):
    print(f'\n{"="*55}')
    print(f'  {label}')
    print(f'{"="*55}')
    preds = load_preds(csv_path)
    m = compute_metrics(preds[['filename', 'top3']], gt[['filename', 'human_topics']])
    print(f'  n={m["n_images"]}  macro-F1={m["macro_f1_mean_per_image"]:.4f}  '
          f'IoU={m["macro_iou_mean_per_image"]:.4f}  '
          f'micro-F1={m["micro_f1"]:.4f}  '
          f'P={m["macro_precision_mean"]:.4f}  R={m["macro_recall_mean"]:.4f}')
    print(f'\n  Per-topic F1:')
    for d in DIMS:
        v = m['per_topic'][d]
        print(f'    {d:<20}: F1={v["f1"]:.3f}  P={v["precision"]:.3f}  R={v["recall"]:.3f}  '
              f'tp={v["tp"]}  fp={v["fp"]}  fn={v["fn"]}')

    # Per-vis-type
    pi = m['per_image_df'].copy()
    gt_vt = gt[['filename', 'VisType']].copy()
    pi = pi.merge(gt_vt, on='filename')
    print(f'\n  Per-vis-type macro-F1 (n≥10):')
    vt_rows = []
    for vt, grp in pi.groupby('VisType'):
        if len(grp) < 10:
            continue
        vt_rows.append((vt, len(grp), grp['f1'].mean()))
        print(f'    {vt:<22}: n={len(grp):>3}  F1={grp["f1"].mean():.3f}')
    return m, vt_rows


if __name__ == '__main__':
    print('Building 510-image topic GT ...')
    gt = build_gt_510()
    print(f'  {len(gt)} images  (human topics/image: median={gt.n_human_topics.median():.0f}  '
          f'mean={gt.n_human_topics.mean():.2f}  range {gt.n_human_topics.min()}-{gt.n_human_topics.max()})')

    results = {}
    for label, dirname in [
        ('V0+T  (510, opus-4.6, t=0)', 'vc_api_510_topicsel_v0_t'),
        ('V0+TW (510, opus-4.6, t=0)', 'vc_api_510_topicsel_v0_tw'),
    ]:
        csv_path = ROOT / 'results' / dirname / 'vc_scores.csv'
        if not csv_path.exists():
            print(f'\n  SKIP {label}: {csv_path} not found')
            continue
        m, vt = run(label, csv_path, gt)
        results[label] = (m, vt)

    if len(results) == 2:
        labels = list(results.keys())
        m1, vt1 = results[labels[0]]
        m2, vt2 = results[labels[1]]
        print(f'\n{"="*55}')
        print('  Summary comparison (510-image vs 63-image pilot)')
        print(f'{"="*55}')
        print(f'{"":22}  {"V0+T":>8}  {"V0+TW":>8}')
        print(f'  {"macro-F1 (510)":22}  {m1["macro_f1_mean_per_image"]:>8.4f}  {m2["macro_f1_mean_per_image"]:>8.4f}')
        print(f'  {"macro-F1 (63, pilot)":22}  {"0.5130":>8}  {"0.5220":>8}')
        print(f'  {"Δ vs pilot":22}  {m1["macro_f1_mean_per_image"]-0.5130:>+8.4f}  {m2["macro_f1_mean_per_image"]-0.5220:>+8.4f}')
        print(f'  {"n images":22}  {m1["n_images"]:>8}  {m2["n_images"]:>8}')

        print(f'\n  Per-topic F1 comparison:')
        print(f'  {"Topic":20}  {"V0+T":>7}  {"V0+TW":>7}  {"Δ(TW-T)":>8}')
        for d in DIMS:
            f1_t  = m1['per_topic'][d]['f1']
            f1_tw = m2['per_topic'][d]['f1']
            print(f'  {d:20}  {f1_t:>7.3f}  {f1_tw:>7.3f}  {f1_tw-f1_t:>+8.3f}')
