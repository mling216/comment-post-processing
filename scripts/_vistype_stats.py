import pandas as pd
df = pd.read_csv('paper/tables/pilot_study_consolidated.csv')
print('Total:', len(df))
print()
print('VisType distribution:')
print(df['VisType'].value_counts().to_string())
print()
print('Anchor images:')
anchors = df[df['isAnchor'] == True][['imageName', 'VisType']]
print(anchors.to_string(index=False))
