import pandas as pd

pr = pd.read_csv('Claude_vc_prediction/pair_images_in_random.csv', nrows=5)
print('=== pair_images_in_random.csv ===')
print('Cols:', pr.columns.tolist())
print(pr.head(3).to_string())

pnr = pd.read_csv('Claude_vc_prediction/pair_images_not_in_random.csv', nrows=5)
print()
print('=== pair_images_not_in_random.csv ===')
print('Cols:', pnr.columns.tolist())
print(pnr.head(3).to_string())

df = pd.read_csv('Claude_vc_prediction/ResultsStepByStep - 0.postquestionare_all.csv')
print()
print('=== ResultsStepByStep shape:', df.shape)
# Check if there's a participant/rater column
non_name_cols = [c for c in df.columns if 'participant' in c.lower() or 'rater' in c.lower() or 'subject' in c.lower() or 'user' in c.lower() or 'worker' in c.lower()]
print('Potential rater cols:', non_name_cols)
# Check the gt_all_46
g46 = pd.read_csv('Claude_vc_prediction/gt_all_46.csv')
print()
print('=== gt_all_46.csv ===')
print('Cols:', g46.columns.tolist())
print('Shape:', g46.shape)
print(g46.head(3).to_string())
