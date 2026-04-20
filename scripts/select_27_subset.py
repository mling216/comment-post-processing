"""
Select 27 images: 3 per VisType (low/mid/high VC) from top 9 VisTypes.
Saves subset CSV for prompt ablation experiments (v0, v1, etc.).

Usage:
    python scripts/select_27_subset.py
"""

import pandas as pd
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
DATA_CSV = Path('phrase_reduction_v2/image_compiled_phrases.csv')
OUT_CSV  = Path('vc_genome_output_full/subset_27.csv')

TOP9_VISTYPES = [
    'Node-link', 'Area', 'Grid', 'Glyph', 'Point',
    'Bar', 'Text', 'Line', 'Cont.-ColorPatn',
]

def main():
    df = pd.read_csv(DATA_CSV)
    df9 = df[df['VisType'].isin(TOP9_VISTYPES)].copy()

    selected = []
    for vt in TOP9_VISTYPES:
        sub = df9[df9['VisType'] == vt].sort_values('NormalizedVC').reset_index(drop=True)
        n = len(sub)
        # low ≈ 10th percentile, mid ≈ 50th, high ≈ 90th
        low_idx  = max(0, int(n * 0.10))
        mid_idx  = n // 2
        high_idx = min(n - 1, int(n * 0.90))

        for label, idx in [('low', low_idx), ('mid', mid_idx), ('high', high_idx)]:
            row = sub.iloc[idx]
            selected.append({
                'imageName':        row['imageName'],
                'imageURL':         f'https://raw.githubusercontent.com/c109363/ExperimentImage/main/AllDataResize/{row["imageName"]}',
                'VisType':          vt,
                'NormalizedVC':     round(row['NormalizedVC'], 4),
                'vc_tier':          label,
                'Topics':           row['Topics'],
                'SubTopics':        row['SubTopics'],
                'rawUserComments':  row['rawUserComments'],
                'originalPhrases':  row['originalPhrases'],
                'humanCuratedPhrases': row['humanCuratedPhrases'],
            })

    result = pd.DataFrame(selected)

    # Print summary
    print(result[['VisType', 'vc_tier', 'NormalizedVC', 'imageName']].to_string(index=False))
    print(f'\nTotal selected: {len(result)}')

    # Save
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)
    print(f'Saved to {OUT_CSV}')


if __name__ == '__main__':
    main()
