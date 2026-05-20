import pandas as pd
import os

def jaccard(a, b):
    sa = set(str(a).split(';')) if pd.notna(a) and str(a).strip() else set()
    sb = set(str(b).split(';')) if pd.notna(b) and str(b).strip() else set()
    if not sa and not sb: return 1.0
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def macro_j(df, pred, gt='gt'):
    return round(df.apply(lambda r: jaccard(r[pred], r[gt]), axis=1).mean(), 3)

def vt_j(df, pred, gt='gt'):
    return {v: round(g.apply(lambda r: jaccard(r[pred], r[gt]), axis=1).mean(), 3)
            for v, g in df.groupby('vistype')}

d63  = pd.read_csv('results/probe_prominent_63/analysis_63.csv')
d510 = pd.read_csv('results/probe_prominent_510_opus_4_6/analysis_510.csv')
d510s= pd.read_csv('results/probe_prominent_510_sonnet-4-5/analysis_510.csv')

print('63 macro: vot=', macro_j(d63,'vot_top3'), 'votw=', macro_j(d63,'votw_top3'), 'prom=', macro_j(d63,'prominent'))
print('510 macro: vot=', macro_j(d510,'vot_top3'), 'votw=', macro_j(d510,'votw_top3'))
print('510 prom opus=', macro_j(d510,'prominent'), 'sonnet=', macro_j(d510s,'prominent'))

for tag, df, col in [
    ('510 V0+T',         d510,  'vot_top3'),
    ('510 V0+TW',        d510,  'votw_top3'),
    ('510 Prom Opus',    d510,  'prominent'),
    ('510 Prom Sonnet',  d510s, 'prominent'),
    ('63 V0+T',          d63,   'vot_top3'),
    ('63 V0+TW',         d63,   'votw_top3'),
    ('63 Prom',          d63,   'prominent'),
]:
    print(f'--- {tag} per vt ---')
    for v, j in sorted(vt_j(df, col).items()):
        print(f'  {v}: {j}')

def jaccard_mean(df, pred_col, gt_col):
    scores = []
    for _, row in df.iterrows():
        gt  = set(str(row[gt_col]).split(';'))  if pd.notna(row[gt_col])  else set()
        pred = set(str(row[pred_col]).split(';')) if pd.notna(row[pred_col]) else set()
        if not gt and not pred:
            scores.append(1.0)
        elif not gt or not pred:
            scores.append(0.0)
        else:
            scores.append(len(gt & pred) / len(gt | pred))
    return round(sum(scores)/len(scores), 3)

# ---- 63-image set ----
folders_63 = {
    'V0+T':        ('vc_api_topicsel_v0_t',         'vot_top3'),
    'V0+TW':       ('vc_api_topicsel_v0_tw',        'votw_top3'),
    'V0+T+VT':     ('vc_api_topicsel_v0_t_vt',      'vot_top3'),
    'V0+TW+VT':    ('vc_api_topicsel_v0_tw_dyn_opus','votw_top3'),
}

print("=== 63-image ===")
for label, (folder, pred_col) in folders_63.items():
    path = f'results/{folder}'
    if not os.path.exists(path):
        print(f'MISSING: {path}'); continue
    csvs = [f for f in os.listdir(path) if f.endswith('.csv')]
    for csv in csvs:
        df = pd.read_csv(f'{path}/{csv}')
        if 'gt' in df.columns and pred_col in df.columns:
            j = jaccard_mean(df, pred_col, 'gt')
            print(f'  {label} [{csv}] Jaccard={j}  n={len(df)}')
            break

# ---- 510-image set ----
folders_510 = {
    'V0+T':        ('vc_api_510_topicsel_v0_t',  'vot_top3'),
    'V0+TW':       ('vc_api_510_topicsel_v0_tw', 'votw_top3'),
    'Prominent (Opus)':  ('probe_prominent_510_opus_4_6',    'prominent'),
    'Prominent (Sonnet)':('probe_prominent_510_sonnet-4-5',  'prominent'),
}

print("=== 510-image ===")
for label, (folder, pred_col) in folders_510.items():
    path = f'results/{folder}'
    if not os.path.exists(path):
        print(f'MISSING: {path}'); continue
    csvs = [f for f in os.listdir(path) if f.endswith('.csv')]
    for csv in csvs:
        df = pd.read_csv(f'{path}/{csv}')
        if 'gt' in df.columns and pred_col in df.columns:
            j = jaccard_mean(df, pred_col, 'gt')
            print(f'  {label} [{csv}] Jaccard={j}  n={len(df)}')
            break

# ---- Prominent 63 ----
print("=== Prominent 63 ===")
path = 'results/probe_prominent_63'
if os.path.exists(path):
    for csv in os.listdir(path):
        if csv.endswith('.csv'):
            df = pd.read_csv(f'{path}/{csv}')
            if 'gt' in df.columns and 'prominent' in df.columns:
                j = jaccard_mean(df, 'prominent', 'gt')
                print(f'  Prominent Opus 63 [{csv}] Jaccard={j}  n={len(df)}')
