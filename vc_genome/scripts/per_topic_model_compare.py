import pandas as pd

topics = ['data_density','visual_encoding','text_annotation','domain_schema',
          'color_symbol','aesthetic_order','cognitive_load']

def per_topic_stats(df):
    rows = {}
    for t in topics:
        gt_bin   = df['gt'].apply(lambda x: int(t in str(x).split(';')))
        pred_bin = df['prominent'].apply(lambda x: int(t in str(x).split(';')))
        tp = (pred_bin & gt_bin).sum()
        p = tp / pred_bin.sum() if pred_bin.sum() else 0.0
        r = tp / gt_bin.sum()   if gt_bin.sum()   else 0.0
        f = 2*p*r/(p+r) if (p+r) else 0.0
        rows[t] = {'P': round(p,3), 'R': round(r,3), 'F1': round(f,3)}
    return rows

opus   = pd.read_csv('results/probe_prominent_510_opus_4_6/analysis_510.csv')
sonnet = pd.read_csv('results/probe_prominent_510_sonnet-4-5/analysis_510.csv')

o = per_topic_stats(opus)
s = per_topic_stats(sonnet)

for t in topics:
    d = round(s[t]['F1'] - o[t]['F1'], 3)
    sign = '+' if d >= 0 else ''
    print(f"{t}: Opus F1={o[t]['F1']} P/R={o[t]['P']}/{o[t]['R']}  "
          f"Sonnet F1={s[t]['F1']} P/R={s[t]['P']}/{s[t]['R']}  delta={sign}{d}")
