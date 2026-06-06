"""
render_oar_B_random_sample.py
==============================
Randomly selects N images from oar_B_510.json that have NOT yet been rendered
in vc_genome/scene_graphs/png/B/, then renders their scene graphs.

Usage:
  python vc_genome/code/render_oar_B_random_sample.py            # default n=15, seed=42
  python vc_genome/code/render_oar_B_random_sample.py --n 20 --seed 99
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

# Locate project root (two levels up from this script)
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT       = SCRIPT_DIR.parent.parent

# Add vc_genome/code to path so we can import render_oar_B
sys.path.insert(0, str(SCRIPT_DIR))
from render_oar_B import render_image, OUT_PNG, OUT_SVG, DATA_CSV

import pandas as pd

DEFAULT_IN = ROOT / 'vc_genome_output_full' / 'three_conditions' / 'oar_B_510.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n',    type=int, default=15, help='Number of images to sample (default 15)')
    ap.add_argument('--seed', type=int, default=42, help='Random seed (default 42)')
    ap.add_argument('--input', type=str, default=None)
    args = ap.parse_args()

    in_path = Path(args.input) if args.input else DEFAULT_IN
    data = json.loads(in_path.read_text(encoding='utf-8'))

    # Images already rendered
    OUT_PNG.mkdir(parents=True, exist_ok=True)
    OUT_SVG.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in OUT_PNG.glob('*.png')}

    available = [k for k in data.keys() if Path(k).stem not in existing]
    if len(available) < args.n:
        print(f'Only {len(available)} images available, sampling all of them.')
    
    random.seed(args.seed)
    selected = random.sample(available, min(args.n, len(available)))

    print(f'Selected {len(selected)} images (seed={args.seed}):')
    for name in selected:
        print(f'  {name}')

    # Load metadata
    meta: dict[str, dict] = {}
    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
        for _, row in df.iterrows():
            meta[row['imageName']] = {
                'VisType':      str(row.get('VisType', '')),
                'NormalizedVC': float(row.get('NormalizedVC', 0)),
            }

    print(f'\nRendering {len(selected)} image(s)...')
    print(f'  SVG → {OUT_SVG.relative_to(ROOT)}')
    print(f'  PNG → {OUT_PNG.relative_to(ROOT)}\n')

    for image_name in selected:
        entry = data[image_name]
        m = meta.get(image_name, {})
        render_image(image_name, entry,
                     vis_type=m.get('VisType', ''),
                     norm_vc=m.get('NormalizedVC', None))

    print(f'\nDone — {len(selected)} image(s) rendered.')
    print('\nSelected image names (for record):')
    for name in selected:
        print(f'  {name}')


if __name__ == '__main__':
    main()
