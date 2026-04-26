"""
Build gt_all_66.csv (66 images = original 46 + 20 new), then compute
per-vistype topic-selection F1 for V0+T and V0+TW on the full 63-image
non-anchor set (7 per vistype × 9 vistypes).

Outputs:
  Claude_vc_prediction/gt_all_66.csv
  topic_selection/topicsel_63_summary.csv
  topic_selection/topicsel_63_per_vistype.csv
  topic_selection/topicsel_63_per_image.csv
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(r'd:\Coding\Copilot\comment_post_processing')
ANCHORS = {'VisC.503.6.png', 'InfoVisJ.619.17.png', 'InfoVisJ.1149.6(1).png'}

TOPIC_TO_DIM = {
    'Data Density / Image Clutter':   'data_density',
    'Visual Encoding Clarity':        'visual_encoding',
    'Semantics / Text Legibility':    'text_annotation',
    'Schema':                         'domain_schema',
    'Color, Symbol, and Texture Details': 'color_symbol',
    'Aesthetics Uncertainty':         'aesthetic_order',
    'Immediacy / Cognitive Load':     'cognitive_load',
}
DIMS = list(TOPIC_TO_DIM.values())

# 20 new images added in the 66-image expansion
NEW_IMAGES = [
    'SciVisJ.259.7.png',
    'SciVisJ.635.6.png',
    'VisC.255.7.png',
    'wsj108.png',
    'InfoVisJ.1933.13.png',
    'SciVisJ.728.14.png',
    'SciVisJ.867.6.png',
    'treasuryG07_2.png',
    'VisC.163.2.png',
    'whoK27_2.png',
    'VASTC.121.5.png',
    'InfoVisJ.485.9.png',
    'economist_daily_chart_152.png',
    'VisC.527.1.png',
    'whoQ32_2.png',
    'economist_daily_chart_393.png',
    'VASTJ.2012.4.png',
    'VASTJ.1763.8(2).png',
    'MorphableWordClouds21.png',
    'InfoVisJ.621.6(2).png',
]

# ── Step 1: Build gt_all_66.csv ──────────────────────────────────────────────
def build_gt66():
    gt46 = pd.read_csv(ROOT / 'Claude_vc_prediction' / 'gt_all_46.csv')
    compiled = pd.read_csv(ROOT / 'phrase_reduction_v2' / 'image_compiled_phrases.csv')

    new_rows = []
    for img in NEW_IMAGES:
        row = compiled[compiled['imageName'] == img]
        if len(row) == 0:
            print(f'  MISSING from compiled phrases: {img}')
            continue
        new_rows.append({'imageName': img, 'NormalizedVC': row.iloc[0]['NormalizedVC']})

    gt_new = pd.DataFrame(new_rows)
    gt66 = pd.concat([gt46, gt_new], ignore_index=True)
    out = ROOT / 'Claude_vc_prediction' / 'gt_all_66.csv'
    gt66.to_csv(out, index=False)
    print(f'Saved gt_all_66.csv ({len(gt66)} images)')
    return gt66


# ── Step 2: Build human GT (topic sets) for all 63 non-anchor images ─────────
def build_human_gt_63(gt66):
    compiled = pd.read_csv(ROOT / 'phrase_reduction_v2' / 'image_compiled_phrases.csv')
    # Also need vistype from compiled
    vistype_map = dict(zip(compiled['imageName'], compiled['VisType']))

    eval_imgs = [n for n in gt66['imageName'].tolist() if n not in ANCHORS]
    assert len(eval_imgs) == 63, f'expected 63 eval imgs, got {len(eval_imgs)}'

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
        for t in topic_labels:
            if t in TOPIC_TO_DIM:
                d = TOPIC_TO_DIM[t]
                if d not in dims:
                    dims.append(d)
        rows.append({
            'filename': img,
            'vistype': vistype_map.get(img, 'Unknown'),
            'n_human_topics': len(dims),
            'human_topics': ';'.join(dims),
        })

    if missing:
        print(f'MISSING from compiled phrases: {missing}')
    return pd.DataFrame(rows)


# ── Step 3: Load V0+T / V0+TW results ────────────────────────────────────────
def load_topicsel(outdir_name: str) -> pd.DataFrame:
    p = ROOT / 'results' / outdir_name / 'vc_scores.csv'
    df = pd.read_csv(p)
    df = df[~df['filename'].isin(ANCHORS)].copy()
    df['top3_topics'] = df['top3_topics'].fillna('')
    return df[['filename', 'top3_topics']].rename(columns={'top3_topics': 'top3'})


# ── Step 4: Compute metrics ───────────────────────────────────────────────────
def set_f1(pred_set, human_set):
    if not pred_set and not human_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not human_set:
        return 0.0, 0.0, 0.0
    tp = len(pred_set & human_set)
    p = tp / len(pred_set)
    r = tp / len(human_set)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def compute_per_vistype(pred_df, gt_df):
    merged = gt_df.merge(pred_df, on='filename', how='left')
    rows = []
    for vt in sorted(merged['vistype'].unique()):
        sub = merged[merged['vistype'] == vt]
        p_list, r_list, f1_list = [], [], []
        for _, row in sub.iterrows():
            human = set(t for t in str(row['human_topics']).split(';') if t)
            pred  = set(t for t in str(row['top3']).split(';') if t)
            p, r, f1 = set_f1(pred, human)
            p_list.append(p); r_list.append(r); f1_list.append(f1)
        rows.append({
            'vistype': vt,
            'n': len(sub),
            'F1': round(sum(f1_list) / len(f1_list), 3),
            'P':  round(sum(p_list)  / len(p_list),  3),
            'R':  round(sum(r_list)  / len(r_list),  3),
        })
    return pd.DataFrame(rows)


def compute_overall(pred_df, gt_df):
    merged = gt_df.merge(pred_df, on='filename', how='left')
    f1s = []
    for _, row in merged.iterrows():
        human = set(t for t in str(row['human_topics']).split(';') if t)
        pred  = set(t for t in str(row['top3']).split(';') if t)
        _, _, f1 = set_f1(pred, human)
        f1s.append(f1)
    return round(sum(f1s) / len(f1s), 3)


if __name__ == '__main__':
    # Step 1: Build 66-image GT
    gt66 = build_gt66()

    # Step 2: Build human GT for 63 non-anchor images
    gt63 = build_human_gt_63(gt66)
    print(f'GT63: {len(gt63)} images, vistypes: {sorted(gt63["vistype"].unique())}')

    # Step 3: Load results
    vot  = load_topicsel('vc_api_topicsel_v0_t')
    votw = load_topicsel('vc_api_topicsel_v0_tw')
    print(f'V0+T  results loaded: {len(vot)} images')
    print(f'V0+TW results loaded: {len(votw)} images')

    # Check if all 63 non-anchor images are scored
    missing_t  = [img for img in gt63['filename'] if img not in vot['filename'].values]
    missing_tw = [img for img in gt63['filename'] if img not in votw['filename'].values]
    print(f'V0+T  missing: {missing_t}')
    print(f'V0+TW missing: {missing_tw}')

    if missing_t or missing_tw:
        print('\nNOTE: Run the scoring scripts on gt_all_66.csv first, then re-run this script.')
        sys.exit(0)

    # Step 4: Compute per-vistype F1
    vt_t  = compute_per_vistype(vot,  gt63)
    vt_tw = compute_per_vistype(votw, gt63)

    print('\nPer-vistype F1 — V0+T:')
    print(vt_t.to_string(index=False))
    print(f'  Overall macro-F1 (image-level): {compute_overall(vot, gt63)}')

    print('\nPer-vistype F1 — V0+TW:')
    print(vt_tw.to_string(index=False))
    print(f'  Overall macro-F1 (image-level): {compute_overall(votw, gt63)}')

    # Step 5: Save outputs
    OUT_DIR = ROOT / 'topic_selection'
    OUT_DIR.mkdir(exist_ok=True)
    vt_t.to_csv(OUT_DIR / 'topicsel_63_per_vistype_vot.csv',  index=False)
    vt_tw.to_csv(OUT_DIR / 'topicsel_63_per_vistype_votw.csv', index=False)

    summary = vt_t.rename(columns={'F1': 'F1_V0T', 'P': 'P_V0T', 'R': 'R_V0T'}).merge(
        vt_tw.rename(columns={'F1': 'F1_V0TW', 'P': 'P_V0TW', 'R': 'R_V0TW'}),
        on=['vistype', 'n']
    )
    summary.to_csv(OUT_DIR / 'topicsel_63_vistype_summary.csv', index=False)
    print(f'\nSaved outputs to {OUT_DIR}')
