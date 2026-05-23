"""
Explore data for human-human CCC and topic agreement ceilings.
"""
import pandas as pd
import numpy as np
import json

# ── 1. Compare gt_all_46 vs gt_all_66 ─────────────────────────────────────────
g46 = pd.read_csv('Claude_vc_prediction/gt_all_46.csv')
g66 = pd.read_csv('Claude_vc_prediction/gt_all_66.csv')
print('=== gt_all_46 vs gt_all_66 ===')
print('46 shape:', g46.shape, '  66 shape:', g66.shape)
merged = g46.merge(g66, on='imageName', suffixes=('_46', '_66'))
print('Overlapping images:', len(merged))
merged['diff'] = merged['NormalizedVC_46'] - merged['NormalizedVC_66']
print('Score differences: max=%.4f mean=%.4f std=%.4f' % (
    merged['diff'].abs().max(), merged['diff'].abs().mean(), merged['diff'].abs().std()))
print(merged[merged['diff'].abs() > 0.01][['imageName', 'NormalizedVC_46', 'NormalizedVC_66']].head(10).to_string())
print()

# ── 2. Per-rater topic data from postquestionnaire ────────────────────────────
df = pd.read_csv('Claude_vc_prediction/ResultsStepByStep - 0.postquestionare_all.csv')
# Build per-image per-rater topic sets
def parse_kw_json(s):
    if pd.isna(s): return set()
    try:
        d = json.loads(str(s))
        return set(d.keys()) if isinstance(d, dict) else set()
    except:
        return set()

# Map old topic names to dim keys (simplified)
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

def kw_to_dims(s):
    raw_topics = parse_kw_json(s)
    dims = set()
    for t in raw_topics:
        for key, dim in TOPIC_TO_DIM.items():
            if key.lower() in t.lower() or t.lower() in key.lower():
                dims.add(dim)
    return dims

# Build per-image sets of (dims, rater_row_index)
image_rater_topics = {}  # imageName -> list of dim sets (one per rater row)
for _, row in df.iterrows():
    for img_col, kw_col in [
        ('MoreComplexImageName', '(Final.MappedBack)HumanCuratedFinalKeywordsMoreComplex'),
        ('LessComplexImageName', '(Final.MappedBack)HumanCuratedFinalKeywordsLessComplex'),
    ]:
        img = row[img_col]
        if pd.isna(img): continue
        dims = kw_to_dims(row[kw_col])
        if not dims: continue
        if img not in image_rater_topics:
            image_rater_topics[img] = []
        image_rater_topics[img].append(dims)

# Coverage stats
counts = [len(v) for v in image_rater_topics.values()]
print('=== Per-image rater coverage (from postquestionnaire) ===')
print('Images with >=1 rater topics:', len(counts))
print('Rater counts: min=%d max=%d mean=%.1f median=%.1f' % (
    min(counts), max(counts), np.mean(counts), np.median(counts)))
print('Images with >=2 raters:', sum(c >= 2 for c in counts))
print('Images with >=3 raters:', sum(c >= 3 for c in counts))
print('Images with >=5 raters:', sum(c >= 5 for c in counts))
print()

# Sample: images with >=3 raters
print('Sample images with >=3 raters:')
sample_imgs = [img for img, v in image_rater_topics.items() if len(v) >= 3][:5]
for img in sample_imgs:
    print(f'  {img}: {len(image_rater_topics[img])} raters → {[sorted(s) for s in image_rater_topics[img]]}')
print()

# ── 3. Check how many of the 63-image GT set have >=2 raters ─────────────────
gt63 = pd.read_csv('Claude_vc_prediction/gt_all_66.csv')
gt63_imgs = set(gt63['imageName'])
gt63_with_raters = {img: v for img, v in image_rater_topics.items() if img in gt63_imgs}
print('=== GT63 coverage ===')
print('GT63 images with >=1 rater topics:', len(gt63_with_raters))
counts63 = [len(v) for v in gt63_with_raters.values()]
print('Rater counts: min=%d max=%d mean=%.1f' % (min(counts63) if counts63 else 0, max(counts63) if counts63 else 0, np.mean(counts63) if counts63 else 0))
print('Images with >=2 raters:', sum(c >= 2 for c in counts63))
print('Images with >=3 raters:', sum(c >= 3 for c in counts63))
