import pandas as pd
df = pd.read_csv('phrase_reduction_v2/image_compiled_phrases.csv')
for img in ['VisC.503.6.png', 'InfoVisJ.1149.6(1).png', 'VASTC.13.9.png']:
    row = df[df['imageName']==img].iloc[0]
    print(f"{img}:")
    print(f"  {row['rawUserComments'][:250]}")
    print()
