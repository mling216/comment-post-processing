import pandas as pd
c = pd.read_csv('phrase_reduction_v2/image_compiled_phrases.csv')
r = c[c['imageName']=='visMost97.png'].iloc[0]
for col in ['Topics', 'SubTopics', 'originalPhrases', 'originalSentiments', 'objectWords', 'actionWords', 'humanCuratedPhrases', 'finalPhrases']:
    print(f'\n=== {col} ===')
    print(str(r[col])[:500])
