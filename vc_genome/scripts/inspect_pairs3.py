"""
Deep inspect of pairwise data for human-human agreement analysis.
"""
import pandas as pd
import numpy as np
import json

df = pd.read_csv('Claude_vc_prediction/ResultsStepByStep - 0.postquestionare_all.csv')
print('=== Pairwise comparison structure ===')
print('diffScore values:', sorted(df['diffScore'].unique()))
print()

# Check each row: are MoreScore/LessScore already aggregate BT scores?
# Look at a few rows where diffScore != 0 (if any)
nonzero = df[df['diffScore'] != 0]
print('Rows with diffScore != 0:', len(nonzero))

# Sample a few rows to understand MoreScore/LessScore meaning
sample = df[['index', 'MoreComplexImageName', 'LessComplexImageName', 
             'moreComplexImageChosen', 'lessComplexImageChosen', 
             'MoreScore', 'LessScore', 'diffScore']].head(10)
print(sample.to_string())

print()
# Check FinalMappedBack keywords - per-rater topic extractions
kw_cols = [c for c in df.columns if 'MappedBack' in c]
print('Keyword cols:', kw_cols)
print()
print('More complex keywords sample (first 3):')
for i, v in enumerate(df[kw_cols[0]].dropna().head(3)):
    print(f'  [{i}]', str(v)[:200])
