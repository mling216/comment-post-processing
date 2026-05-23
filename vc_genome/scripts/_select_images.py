import pandas as pd

compiled = pd.read_csv('phrase_reduction_v2/image_compiled_phrases.csv')
pairs_df = pd.read_csv('GPT_image_pair_comparison/image_pairs_results.csv')
pair_imgs = set(pairs_df['image1'].tolist() + pairs_df['image2'].tolist())
print(f'Total pair images: {len(pair_imgs)}')

pair_compiled = compiled[compiled['imageName'].isin(pair_imgs)].copy()
print(f'Matched in compiled: {len(pair_compiled)}')

print('\nVisType distribution in 46 pair images:')
for vt, grp in pair_compiled.groupby('VisType'):
    imgs = grp.sort_values('NormalizedVC')
    print(f'\n{vt} ({len(grp)} images):')
    for _, r in imgs.iterrows():
        print(f'  {r["imageName"]:40s} VC={r["NormalizedVC"]:.2f}')

print('\n\n=== SELECTED (1 low + 1 high per vistype) ===')
selected = []
for vt, grp in pair_compiled.groupby('VisType'):
    sorted_g = grp.sort_values('NormalizedVC')
    low = sorted_g.iloc[0]
    high = sorted_g.iloc[-1]
    if len(grp) >= 2 and low['imageName'] != high['imageName']:
        selected.append(low['imageName'])
        selected.append(high['imageName'])
        print(f'{vt}: {low["imageName"]} (VC={low["NormalizedVC"]:.2f}), {high["imageName"]} (VC={high["NormalizedVC"]:.2f})')
    else:
        selected.append(low['imageName'])
        print(f'{vt}: {low["imageName"]} (VC={low["NormalizedVC"]:.2f})')
print(f'\nTotal selected: {len(selected)}')
