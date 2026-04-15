import pandas as pd

compiled = pd.read_csv('phrase_reduction_v2/image_compiled_phrases.csv')
pairs_df = pd.read_csv('GPT_image_pair_comparison/image_pairs_results.csv')

pair_imgs = set(pairs_df['image1'].tolist() + pairs_df['image2'].tolist())
pair_compiled = compiled[compiled['imageName'].isin(pair_imgs)].copy()

selected = []
for vt, grp in pair_compiled.groupby('VisType'):
    sorted_g = grp.sort_values('NormalizedVC')
    selected.append(sorted_g.iloc[0]['imageName'])
    if len(grp) >= 2:
        selected.append(sorted_g.iloc[-1]['imageName'])

examples = compiled[compiled['imageName'].isin(selected)].copy()
examples = examples.set_index('imageName').loc[selected].reset_index()

for _, row in examples.iterrows():
    print(f"===== {row['imageName']} | {row['VisType']} | VC={row['NormalizedVC']:.2f} =====")
    print(f"objectWords: {row['objectWords']}")
    print(f"actionWords: {row['actionWords']}")
    print(f"SubTopics: {row['SubTopics']}")
    print(f"originalPhrases: {row['originalPhrases']}")
    print(f"originalSentiments: {row['originalSentiments']}")
    print(f"rawUserComments: {row['rawUserComments']}")
    print(f"humanCuratedPhrases: {row['humanCuratedPhrases']}")
    print()
