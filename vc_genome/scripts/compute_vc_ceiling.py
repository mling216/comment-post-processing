"""
Compute human-human VC CCC ceiling via split-half Bradley-Terry.

Each row in the postquestionnaire CSV is a pairwise judgment:
  MoreComplexImageName > LessComplexImageName  (winner > loser)

Method:
  1. Fit Bradley-Terry on all comparisons -> full BT score vector
  2. Split comparisons randomly into two halves; fit BT on each half
  3. Match images present in both halves; compute CCC between the two score vectors
  4. Repeat n_splits=500 times; report mean CCC as inter-rater reliability ceiling
  5. Also report CCC(full BT, NormalizedVC GT) as sanity check

Run from: d:\Coding\Copilot\comment_post_processing\
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats, optimize

ROOT    = Path('.')
RESULTS = ROOT / 'results'
ANCHORS = {'VisC.503.6.png', 'InfoVisJ.619.17.png', 'InfoVisJ.1149.6(1).png'}

# ─── Bradley-Terry via Zermelo's iterative algorithm ─────────────────────────
def fit_bt(pairs: list[tuple[str, str]], max_iter: int = 100, tol: float = 1e-6) -> dict[str, float]:
    """
    Vectorized Bradley-Terry via Zermelo algorithm.
    pairs: list of (winner, loser) tuples.
    Returns dict {imageName: BT_score}, mean-normalized.
    """
    images = sorted(set(w for w, _ in pairs) | set(l for _, l in pairs))
    idx    = {img: i for i, img in enumerate(images)}
    n      = len(images)

    w_idx = np.array([idx[w] for w, _ in pairs])
    l_idx = np.array([idx[l] for _, l in pairs])
    wins  = np.bincount(w_idx, minlength=n).astype(float)

    s = np.ones(n)
    for _ in range(max_iter):
        denom_pair = s[w_idx] + s[l_idx]
        contrib    = 1.0 / denom_pair
        sum_contrib = np.zeros(n)
        np.add.at(sum_contrib, w_idx, contrib)
        np.add.at(sum_contrib, l_idx, contrib)
        with np.errstate(divide='ignore', invalid='ignore'):
            s_new = np.where(sum_contrib > 0, wins / sum_contrib, 0.0)
        mean_s = s_new.mean()
        if mean_s > 0:
            s_new /= mean_s
        if np.max(np.abs(s_new - s)) < tol:
            s = s_new
            break
        s = s_new

    return {img: float(s[idx[img]]) for img in images}


def ccc(y: np.ndarray, yh: np.ndarray) -> float:
    n    = len(y)
    mu_y, mu_h = y.mean(), yh.mean()
    s2_y = y.var()
    s2_h = yh.var()
    cov  = ((y - mu_y) * (yh - mu_h)).mean()
    denom = s2_y + s2_h + (mu_y - mu_h) ** 2
    return float(2 * cov / denom) if denom != 0 else 0.0


def main():
    print('=' * 65)
    print('  VC SCORING — Human-Human CCC Ceiling (Split-Half BT)')
    print('=' * 65)

    # ── Load comparisons ────────────────────────────────────────────────────
    df = pd.read_csv(ROOT / 'Claude_vc_prediction/ResultsStepByStep - 0.postquestionare_all.csv')

    # Each row: MoreComplexImageName won against LessComplexImageName
    pairs_all = [
        (str(row['MoreComplexImageName']).strip(), str(row['LessComplexImageName']).strip())
        for _, row in df.iterrows()
        if pd.notna(row['MoreComplexImageName']) and pd.notna(row['LessComplexImageName'])
        and str(row['MoreComplexImageName']).strip() not in ANCHORS
        and str(row['LessComplexImageName']).strip() not in ANCHORS
    ]
    print(f'  Total valid comparisons: {len(pairs_all)}')

    all_imgs = set(w for w, _ in pairs_all) | set(l for _, l in pairs_all)
    print(f'  Unique images: {len(all_imgs)}')

    # ── Load GT NormalizedVC ────────────────────────────────────────────────
    gt = pd.read_csv(ROOT / 'Claude_vc_prediction/gt_all_66.csv')
    gt_dict = dict(zip(gt['imageName'], gt['NormalizedVC']))
    gt_in_pairs = {img: gt_dict[img] for img in all_imgs if img in gt_dict}
    print(f'  GT63 images with comparisons: {len(gt_in_pairs)}')
    print()

    # ── Fit BT on all comparisons ────────────────────────────────────────────
    bt_full = fit_bt(pairs_all)

    # CCC between full BT and NormalizedVC GT (z-score both to remove scale mismatch)
    common = sorted(gt_in_pairs.keys())
    y_gt   = np.array([gt_in_pairs[img] for img in common])
    y_bt   = np.array([bt_full[img] for img in common])
    # z-score for scale-free comparison
    y_gt_z = (y_gt - y_gt.mean()) / y_gt.std()
    y_bt_z = (y_bt - y_bt.mean()) / y_bt.std()
    ccc_bt_gt = ccc(y_gt_z, y_bt_z)
    rho_bt_gt = float(stats.spearmanr(y_gt, y_bt)[0])
    print(f'  Full-BT vs GT NormalizedVC (n={len(common)}):')
    print(f'    CCC={ccc_bt_gt:.4f}  rho={rho_bt_gt:.4f}')
    print()

    # ── Split-half BT CCC ────────────────────────────────────────────────────
    n_splits = 200
    rng      = np.random.default_rng(42)
    pairs_arr = np.array(pairs_all, dtype=object)
    n_pairs   = len(pairs_arr)

    ccc_vals = []
    rho_vals = []
    n_used   = []

    for _ in range(n_splits):
        perm   = rng.permutation(n_pairs)
        half1  = [tuple(p) for p in pairs_arr[perm[:n_pairs // 2]]]
        half2  = [tuple(p) for p in pairs_arr[perm[n_pairs // 2:]]]

        bt1 = fit_bt(half1)
        bt2 = fit_bt(half2)

        common_half = sorted(set(bt1) & set(bt2))
        if len(common_half) < 10:
            continue

        v1 = np.array([bt1[img] for img in common_half])
        v2 = np.array([bt2[img] for img in common_half])
        ccc_vals.append(ccc(v1, v2))
        rho_vals.append(float(stats.spearmanr(v1, v2)[0]))
        n_used.append(len(common_half))

    print(f'  Split-half CCC ({n_splits} random splits):')
    print(f'    Mean CCC:             {np.mean(ccc_vals):.4f}')
    print(f'    Std CCC:              {np.std(ccc_vals):.4f}')
    print(f'    95% range:            [{np.percentile(ccc_vals, 2.5):.4f}, {np.percentile(ccc_vals, 97.5):.4f}]')
    print()
    print(f'  Split-half Spearman rho ({n_splits} splits):')
    print(f'    Mean rho:             {np.mean(rho_vals):.4f}')
    print(f'    95% range:            [{np.percentile(rho_vals, 2.5):.4f}, {np.percentile(rho_vals, 97.5):.4f}]')
    print()
    print(f'  Mean images per half (in common): {np.mean(n_used):.0f}')
    print()

    # ── Spearman-Brown corrected ceiling ─────────────────────────────────────
    # Split-half CCC uses K=0.5 of data. Correct to full-data ceiling via SB:
    #   r_full = 2*r_half / (1 + r_half)
    r_half_ccc = np.mean(ccc_vals)
    r_half_rho = np.mean(rho_vals)
    r_sb_ccc   = 2 * r_half_ccc / (1 + r_half_ccc)
    r_sb_rho   = 2 * r_half_rho / (1 + r_half_rho)
    print(f'  Spearman-Brown corrected ceiling (half->full):')
    print(f'    r_SB (CCC-based): {r_sb_ccc:.4f}')
    print(f'    r_SB (rho-based): {r_sb_rho:.4f}  <- preferred (scale-agnostic)')
    print()
    print(f'  NOTE: Mean comparisons/image = {len(pairs_all)/len(all_imgs):.1f}')
    print(f'  Half-split gives ~{len(pairs_all)/2/len(all_imgs):.1f} comp/image.')
    print(f'  Only {int(np.mean(n_used))} of {len(all_imgs)} images appear in both halves.')
    print(f'  With < 1 comp/image per half, BT scores are extremely noisy.')
    print(f'  This data (postquestionnaire, ~1 comp/image) cannot produce a')
    print(f'  stable inter-rater ceiling. The original study database with')
    print(f'  more comparisons per image is needed.')
    print()

    # ── Compare LLM vs ceiling ────────────────────────────────────────────────
    print('  LLM CCC on GT63 set vs rho-based SB ceiling (for reference):')
    try:
        bstrap = pd.read_csv(RESULTS / 'alignment_vc_bootstrap.csv')
        for _, row in bstrap[bstrap['variant'].str.contains('63')].iterrows():
            margin = (row['spearman_rho'] / r_sb_rho) * 100
            print(f'    {row["variant"]:<22} rho={row["spearman_rho"]:.4f}  '
                  f'vs rho-SB-ceiling {r_sb_rho:.4f}: {margin:.0f}%')
        print()
        print(f'  NOTE: LLM rho >> SB ceiling because the postquestionnaire data')
        print(f'  is too sparse. DO NOT report this as "LLM exceeds ceiling".')
    except FileNotFoundError:
        print('    (alignment_vc_bootstrap.csv not found)')
    print()

    # ── Save ──────────────────────────────────────────────────────────────────
    out = pd.DataFrame({
        'metric': ['split_half_CCC_mean', 'split_half_CCC_std',
                   'split_half_CCC_lo95', 'split_half_CCC_hi95',
                   'split_half_rho_mean', 'split_half_rho_lo95', 'split_half_rho_hi95',
                   'SB_ceiling_CCC_based', 'SB_ceiling_rho_based',
                   'full_BT_vs_GT_rho',
                   'mean_comps_per_image', 'mean_images_per_half',
                   'n_comparisons', 'n_images_total', 'n_gt63_in_pairs'],
        'value': [round(np.mean(ccc_vals), 4), round(np.std(ccc_vals), 4),
                  round(np.percentile(ccc_vals, 2.5), 4), round(np.percentile(ccc_vals, 97.5), 4),
                  round(np.mean(rho_vals), 4),
                  round(np.percentile(rho_vals, 2.5), 4), round(np.percentile(rho_vals, 97.5), 4),
                  round(r_sb_ccc, 4), round(r_sb_rho, 4),
                  round(rho_bt_gt, 4),
                  round(len(pairs_all)/len(all_imgs), 2), round(np.mean(n_used), 1),
                  len(pairs_all), len(all_imgs), len(gt_in_pairs)],
    })
    out_path = RESULTS / 'alignment_vc_ceiling.csv'
    out.to_csv(out_path, index=False)
    print(f'  Saved -> {out_path}')
    print()
    print('Done.')


if __name__ == '__main__':
    main()
