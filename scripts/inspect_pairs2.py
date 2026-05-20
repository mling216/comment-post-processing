"""
Check pairwise data structure and assess feasibility of split-half human-human CCC.
"""
import pandas as pd

df = pd.read_csv('Claude_vc_prediction/ResultsStepByStep - 0.postquestionare_all.csv')
print('Shape:', df.shape)
print('moreChosen notna:', df['moreComplexImageChosen'].notna().sum())
print('lessChosen notna:', df['lessComplexImageChosen'].notna().sum())
print('moreChosen sample:', df['moreComplexImageChosen'].dropna().head(3).tolist())
print()

all_imgs = set(df['MoreComplexImageName'].dropna()) | set(df['LessComplexImageName'].dropna())
print('Unique images in pair comparisons:', len(all_imgs))

mc = df['MoreComplexImageName'].value_counts()
lc = df['LessComplexImageName'].value_counts()
combined = mc.add(lc, fill_value=0)
print('Comparisons/image: min=%d max=%d mean=%.1f' % (combined.min(), combined.max(), combined.mean()))
print('Images with >=5 comps:', (combined >= 5).sum())
print('Images with >=10 comps:', (combined >= 10).sum())
print()

# Check overlap with GT 63-image set
gt63 = pd.read_csv('Claude_vc_prediction/gt_all_66.csv')
print('GT63 images:', len(gt63))
overlap = set(gt63['imageName']) & all_imgs
print('GT63 images in pairwise data:', len(overlap))

# Check the vc_api_1800_v0_tw_input.csv structure
inp1800 = pd.read_csv('results/vc_api_1800_v0_tw_input.csv')
print()
print('=== vc_api_1800_v0_tw_input.csv ===')
print('Cols:', inp1800.columns.tolist())
print('Shape:', inp1800.shape)
print(inp1800.head(3).to_string())
