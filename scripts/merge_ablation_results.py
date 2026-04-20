"""
Merge ablation extraction results with subset_27 metadata.
Adds v0 topics + explanation columns and saves to CSV.

Usage:
    python scripts/merge_ablation_results.py
"""

import json
import pandas as pd
from pathlib import Path


def merge_v0(subset, ablation_dir):
    v0_file = ablation_dir / 'llm_extractions_v0.json'
    if not v0_file.exists():
        print(f'  Skipping v0: {v0_file} not found')
        return subset

    v0 = json.loads(v0_file.read_text(encoding='utf-8'))
    topics_list = []
    explanations = []
    for _, row in subset.iterrows():
        ext = v0.get(row['imageName'], {})
        t = [x.get('topic', '?') for x in ext.get('topics', [])]
        topics_list.append('; '.join(t))
        explanations.append(ext.get('explanation', ''))

    subset['v0_topics'] = topics_list
    subset['v0_explanation'] = explanations
    return subset


def main():
    subset_csv = Path('vc_genome_output_full/subset_27.csv')
    ablation_dir = Path('vc_genome_output_full/ablation')
    out_csv = ablation_dir / 'subset_27_with_results.csv'

    subset = pd.read_csv(subset_csv)
    print(f'Loaded {len(subset)} images from {subset_csv}')

    subset = merge_v0(subset, ablation_dir)

    subset.to_csv(out_csv, index=False)
    print(f'Saved to {out_csv}')
    print(subset[['imageName', 'VisType', 'vc_tier', 'NormalizedVC', 'v0_topics']].to_string(index=False))


if __name__ == '__main__':
    main()
