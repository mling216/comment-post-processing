"""
Compute human-human agreement ceilings:
  Part A: Topic selection — pairwise inter-rater Jaccard (and F1) on 63-image set
  Part B: VC scoring — report what's computable vs. what's missing

Run from: d:\Coding\Copilot\comment_post_processing\
"""
import pandas as pd
import numpy as np
import json
import itertools
from pathlib import Path
from scipy import stats

ROOT = Path('.')
RESULTS = ROOT / 'results'

# ─── Topic name → dim key mapping ────────────────────────────────────────────
TOPIC_TO_DIM = {
    'Data Density / Image Clutter': 'data_density',
    'Density / Clutter': 'data_density',
    'Visual Encoding Clarity': 'visual_encoding',
    'Encoding Clarity': 'visual_encoding',
    'Semantics / Text Legibility': 'text_annotation',
    'Text Legibility': 'text_annotation',
    'Schema': 'domain_schema',
    'Domain Knowledge Requirements': 'domain_schema',
    'Color, Symbol, and Texture Details': 'color_symbol',
    'Color Details': 'color_symbol',
    'Aesthetics Uncertainty': 'aesthetic_order',
    'Aesthetics': 'aesthetic_order',
    'Immediacy / Cognitive Load': 'cognitive_load',
    'Cognitive Load': 'cognitive_load',
}

def parse_kw_json(s):
    if pd.isna(s):
        return set()
    try:
        d = json.loads(str(s))
        return set(d.keys()) if isinstance(d, dict) else set()
    except Exception:
        return set()

def kw_to_dims(s):
    raw_topics = parse_kw_json(s)
    dims = set()
    for t in raw_topics:
        for key, dim in TOPIC_TO_DIM.items():
            if key.lower() in t.lower() or t.lower() in key.lower():
                dims.add(dim)
    return dims

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0

def f1_set(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    tp = len(a & b)
    p = tp / len(a)
    r = tp / len(b)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

# ─── Load per-rater topic data from postquestionnaire ────────────────────────
def load_per_rater_topics():
    """
    Returns: dict { imageName -> list of dim sets (one per rater) }
    Each rater row contributes a topic set for each image it appeared in.
    """
    df = pd.read_csv(
        ROOT / 'Claude_vc_prediction/ResultsStepByStep - 0.postquestionare_all.csv'
    )
    image_rater_topics = {}
    kw_pairs = [
        ('MoreComplexImageName',  '(Final.MappedBack)HumanCuratedFinalKeywordsMoreComplex'),
        ('LessComplexImageName',  '(Final.MappedBack)HumanCuratedFinalKeywordsLessComplex'),
    ]
    for _, row in df.iterrows():
        for img_col, kw_col in kw_pairs:
            img = row[img_col]
            if pd.isna(img):
                continue
            dims = kw_to_dims(row[kw_col])
            if not dims:
                continue
            if img not in image_rater_topics:
                image_rater_topics[img] = []
            image_rater_topics[img].append(dims)
    return image_rater_topics


# ─── Compute pairwise inter-rater agreement per image ────────────────────────
def inter_rater_agreement(image_rater_topics, min_raters=2):
    """
    For each image with >= min_raters raters, compute all pairwise Jaccard and F1.
    Returns DataFrame with per-image mean pairwise metrics.
    """
    rows = []
    for img, rater_sets in image_rater_topics.items():
        if len(rater_sets) < min_raters:
            continue
        pair_jacc = []
        pair_f1 = []
        for a, b in itertools.combinations(rater_sets, 2):
            pair_jacc.append(jaccard(a, b))
            # symmetric F1
            pair_f1.append(f1_set(a, b))
        rows.append(dict(
            imageName=img,
            n_raters=len(rater_sets),
            n_pairs=len(pair_jacc),
            mean_pairwise_jaccard=np.mean(pair_jacc),
            mean_pairwise_f1=np.mean(pair_f1),
            min_pairwise_jaccard=np.min(pair_jacc),
        ))
    return pd.DataFrame(rows)


# ─── Aggregate union GT vs individual rater agreement ────────────────────────
def union_gt_agreement(image_rater_topics, min_raters=2):
    """
    Compute: for each image, union of all rater topic sets = 'aggregate GT'.
    Then compute each rater's Jaccard/F1 against that union GT.
    This mirrors: LLM vs human-aggregate comparison, but for human vs human-aggregate.
    """
    rows = []
    for img, rater_sets in image_rater_topics.items():
        if len(rater_sets) < min_raters:
            continue
        union_gt = set().union(*rater_sets)
        per_rater_jacc = [jaccard(s, union_gt) for s in rater_sets]
        per_rater_f1   = [f1_set(s, union_gt)  for s in rater_sets]
        rows.append(dict(
            imageName=img,
            n_raters=len(rater_sets),
            n_dims_union=len(union_gt),
            mean_rater_jaccard=np.mean(per_rater_jacc),
            mean_rater_f1=np.mean(per_rater_f1),
        ))
    return pd.DataFrame(rows)


# ─── VC CCC ceiling assessment ────────────────────────────────────────────────
def assess_vc_ceiling():
    """
    Check what's available for VC human-human CCC ceiling.
    Returns a dict with diagnostic info.
    """
    df = pd.read_csv(
        ROOT / 'Claude_vc_prediction/ResultsStepByStep - 0.postquestionare_all.csv'
    )
    choice_notna = df['moreComplexImageChosen'].notna().sum()
    g46 = pd.read_csv(ROOT / 'Claude_vc_prediction/gt_all_46.csv')
    g66 = pd.read_csv(ROOT / 'Claude_vc_prediction/gt_all_66.csv')
    merged = g46.merge(g66, on='imageName', suffixes=('_46', '_66'))
    max_diff = (merged['NormalizedVC_46'] - merged['NormalizedVC_66']).abs().max()
    return dict(
        pair_rows=len(df),
        binary_choice_available=(choice_notna > 0),
        gt46_n=len(g46), gt66_n=len(g66),
        gt46_in_gt66=len(merged),
        gt46_gt66_identical=(max_diff == 0.0),
    )


def main():
    print('=' * 65)
    print('  HUMAN–HUMAN AGREEMENT CEILING ANALYSIS')
    print('=' * 65)

    # ── Part A: VC ceiling diagnostic ──────────────────────────────────────
    print()
    print('PART A — VC Scoring: ceiling data availability')
    print('-' * 65)
    vc_info = assess_vc_ceiling()
    print(f'  Pair comparison rows in postquestionnaire CSV: {vc_info["pair_rows"]}')
    print(f'  Binary choice columns (moreComplexImageChosen) available: {vc_info["binary_choice_available"]}')
    print(f'  gt_all_46 (n={vc_info["gt46_n"]}) is subset of gt_all_66 (n={vc_info["gt66_n"]}): '
          f'{vc_info["gt46_in_gt66"]} overlapping images')
    print(f'  gt_all_46 scores identical to gt_all_66: {vc_info["gt46_gt66_identical"]}')
    print()
    print('  CONCLUSION: Raw binary pairwise choice data not available in')
    print('  accessible CSV exports. Split-half Bradley-Terry CCC cannot be')
    print('  computed without the original experiment database.')
    print()
    print('  What WOULD be needed:')
    print('    - Per-participant win/loss judgments: (participant_id, imageA, imageB, winner)')
    print('    - Split participants into two groups; run BT on each half')
    print('    - Compute CCC between two half-BT score vectors')
    print('    - Bootstrap this split many times; report mean CCC as ceiling')
    print()
    print('  Practical alternative (Spearman-Brown): if inter-rater CCC for')
    print('  a single comparison round is ~r_1, and there are K rounds,')
    print('    CCC_full = K*r_1 / (1 + (K-1)*r_1)')
    print('  But r_1 requires the raw data above.')

    # ── Part B: Topic selection ceiling ────────────────────────────────────
    print()
    print('=' * 65)
    print('  PART B — Topic Selection: human-human inter-rater Jaccard')
    print('=' * 65)

    image_rater_topics = load_per_rater_topics()

    # Full dataset stats
    counts = [len(v) for v in image_rater_topics.values()]
    print(f'  Total images with >= 1 rater topics: {len(counts)}')
    print(f'  Raters/image: min={min(counts)} max={max(counts)} '
          f'mean={np.mean(counts):.1f} median={np.median(counts):.1f}')
    print(f'  Images with >= 2 raters: {sum(c >= 2 for c in counts)}')
    print(f'  Images with >= 3 raters: {sum(c >= 3 for c in counts)}')
    print()

    # Pairwise inter-rater agreement (all images with >=2 raters)
    print('  --- Pairwise inter-rater Jaccard (all images >= 2 raters) ---')
    df_pair = inter_rater_agreement(image_rater_topics, min_raters=2)
    print(f'  Images in analysis: {len(df_pair)}')
    print(f'  Total rater-pairs: {df_pair["n_pairs"].sum()}')
    print(f'  Mean pairwise Jaccard: {df_pair["mean_pairwise_jaccard"].mean():.4f}')
    print(f'  Mean pairwise F1:      {df_pair["mean_pairwise_f1"].mean():.4f}')
    print()

    # Restrict to GT63 images
    gt63 = pd.read_csv(ROOT / 'Claude_vc_prediction/gt_all_66.csv')
    gt63_imgs = set(gt63['imageName'])
    gt63_rater_topics = {k: v for k, v in image_rater_topics.items() if k in gt63_imgs}
    counts63 = [len(v) for v in gt63_rater_topics.values()]
    print('  --- GT63-image subset ---')
    print(f'  GT63 images with >= 1 rater topics: {len(counts63)}')
    print(f'  Raters/image: min={min(counts63)} max={max(counts63)} '
          f'mean={np.mean(counts63):.1f}')

    df_pair63 = inter_rater_agreement(gt63_rater_topics, min_raters=2)
    print(f'  Images with >= 2 raters: {len(df_pair63)} '
          f'({len(df_pair63)/len(gt63)*100:.0f}% of GT63)')
    if len(df_pair63) > 0:
        print(f'  Mean pairwise Jaccard: {df_pair63["mean_pairwise_jaccard"].mean():.4f}')
        print(f'  Mean pairwise F1:      {df_pair63["mean_pairwise_f1"].mean():.4f}')
        print(f'  Weighted by n_pairs:')
        wj = np.average(df_pair63['mean_pairwise_jaccard'],
                        weights=df_pair63['n_pairs'])
        wf = np.average(df_pair63['mean_pairwise_f1'],
                        weights=df_pair63['n_pairs'])
        print(f'    Weighted-mean Jaccard: {wj:.4f}')
        print(f'    Weighted-mean F1:      {wf:.4f}')
    print()

    # Union-GT ceiling (mirrors how LLM is evaluated)
    df_union63 = union_gt_agreement(gt63_rater_topics, min_raters=2)
    print('  --- Individual rater vs. union-GT (mirrors LLM evaluation) ---')
    print(f'  Images: {len(df_union63)}')
    if len(df_union63) > 0:
        print(f'  Mean rater-vs-union Jaccard: {df_union63["mean_rater_jaccard"].mean():.4f}')
        print(f'  Mean rater-vs-union F1:      {df_union63["mean_rater_f1"].mean():.4f}')
    print()

    # ── Compare with LLM performance on overlapping subset ─────────────────
    print('  --- LLM vs same-image subset (fair comparison) ---')
    # Load LLM predictions for overlapping images
    llm_pred_csv = RESULTS / 'vc_api_topicsel_v0_t/vc_scores.csv'
    if llm_pred_csv.exists():
        llm_df = pd.read_csv(llm_pred_csv)
        fname_col = 'filename' if 'filename' in llm_df.columns else 'imageName'
        if 'top3_topics' in llm_df.columns:
            llm_topics = {
                row[fname_col]: set(t.strip() for t in str(row['top3_topics']).split(';') if t.strip())
                for _, row in llm_df.iterrows()
            }
            imgs_both = [img for img in df_pair63['imageName'] if img in llm_topics]
            # Compare: LLM vs union-GT  vs  human-individual vs union-GT (same images)
            jacc_llm   = []
            jacc_hv_union = []  # individual human vs union-GT (ceiling)
            jacc_hh_pair  = []  # pairwise human-human (agreement floor)
            for img in imgs_both:
                union_gt = set().union(*gt63_rater_topics[img])
                jacc_llm.append(jaccard(llm_topics.get(img, set()), union_gt))
                # individual rater vs union-GT (average across raters)
                per_rater = [jaccard(s, union_gt) for s in gt63_rater_topics[img]]
                jacc_hv_union.append(np.mean(per_rater))
                # mean pairwise human-human Jaccard
                row = df_pair63[df_pair63['imageName'] == img]
                if len(row):
                    jacc_hh_pair.append(row.iloc[0]['mean_pairwise_jaccard'])
            if jacc_llm:
                print(f'  Overlap images (LLM + >=2 human raters): {len(imgs_both)}')
                llm_j = np.mean(jacc_llm)
                hv_j  = np.mean(jacc_hv_union)   # ceiling: human-vs-union
                hh_j  = np.mean(jacc_hh_pair)    # human-human pairwise agreement
                print(f'  LLM Jaccard vs union-GT:                 {llm_j:.4f}')
                print(f'  Human ceiling (indiv vs union-GT):       {hv_j:.4f}')
                print(f'  Human-human pairwise Jaccard:            {hh_j:.4f}')
                ratio = llm_j / hv_j if hv_j > 0 else float('nan')
                print(f'  LLM / ceiling ratio:                     {ratio:.3f}')
                print()
                print('  NOTE: LLM is evaluated against union-GT (same as LLM evaluation mode).')
                print('  The ceiling (individual human vs union-GT) is the right comparator.')
                print('  Pairwise H-H Jaccard = agreement between any two individual raters,')
                print('  which is lower than human-vs-aggregate and a floor, not ceiling.')
        else:
            print('  (top3_topics column not found in LLM pred file)')
    else:
        print(f'  LLM pred file not found: {llm_pred_csv}')

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = RESULTS / 'alignment_human_ceiling.csv'
    df_union63.to_csv(out_path, index=False)
    print()
    print(f'  Saved per-image union-GT data -> {out_path}')
    print()
    print('Done.')


if __name__ == '__main__':
    main()
