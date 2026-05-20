import pandas as pd, json
df = pd.read_csv('phrase_reduction_v2/image_compiled_phrases.csv')
nine = ['Area','Bar','Cont.-ColorPatn','Glyph','Grid','Line','Node-link','Point','Text']
df9 = df[df.VisType.isin(nine)]
print('9 vistypes total:', len(df9))
print(df9['VisType'].value_counts().to_string())

oar_b  = json.loads(open('vc_genome_output_full/three_conditions/oar_B.json').read())
oar_v1 = json.loads(open('vc_genome_output_full/three_conditions/oar_V1.json').read())
print(f'\nExisting oar_B: {len(oar_b)}  oar_V1: {len(oar_v1)}')
print(f'oar_B images in df9: {len(set(oar_b) & set(df9.imageName))}')
print(f'oar_V1 images in df9: {len(set(oar_v1) & set(df9.imageName))}')
print(f'df9 NOT in oar_B: {len(set(df9.imageName) - set(oar_b))}')
print(f'df9 NOT in oar_V1: {len(set(df9.imageName) - set(oar_v1))}')
