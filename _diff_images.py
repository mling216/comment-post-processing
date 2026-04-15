import pandas as pd

# 1. Get all unique image names from pairs xlsx
pairs = pd.read_excel(r'd:\Coding\Copilot\comment_post_processing\Claude_image_pair_comparison\image_pairs.xlsx')
all_pair_images = set(pairs['image1'].tolist() + pairs['image2'].tolist())
print(f'Total unique images in pairs: {len(all_pair_images)}')

# 2. Get images in random_images.csv
rand = pd.read_csv(r'D:\Coding\Image_VC\random_images.csv')
random_images = set(rand['filename'].tolist())
print(f'Images in random_images: {len(random_images)}')

# 3. Find images in pairs but NOT in random
diff = all_pair_images - random_images
print(f'Images in pairs but not in random: {len(diff)}')

# 4. Get VC scores
vc = pd.read_csv(r'd:\Coding\Copilot\comment_post_processing\phrase_reduction_v2\image_compiled_phrases.csv')[['imageName','NormalizedVC']].drop_duplicates()
result = pd.DataFrame({'imageName': sorted(diff)})
result = result.merge(vc, on='imageName', how='left')
result = result.sort_values('NormalizedVC').reset_index(drop=True)

# Check for missing VC scores
missing = result[result['NormalizedVC'].isna()]
if len(missing) > 0:
    print(f'WARNING: {len(missing)} images have no VC score: {missing["imageName"].tolist()}')

print(result.to_string())

# 5. Save
out = r'd:\Coding\Copilot\comment_post_processing\Claude_image_pair_comparison\pair_images_not_in_random.csv'
result.to_csv(out, index=False)
print(f'\nSaved to {out}')
