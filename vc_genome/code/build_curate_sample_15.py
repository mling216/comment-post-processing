"""
build_curate_sample_15.py
--------------------------
Creates curate_dict_15_sample_oar.csv with the same structure as
curate_dict_9_sample_oar.csv, populated with the 15 newly rendered
B-condition scene graph images.
"""
import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent.parent
MAIN    = ROOT / 'comment_process' / 'ResultsStepByStep - 4.0.imageDataCompiled.csv'
OUT     = ROOT / 'vc_genome' / 'export' / 'curate_dict_15_sample_oar.csv'

IMAGES = [
    'v484_n7392_1_f4.png',
    'economist_daily_chart_152.png',
    'VisJ.1515.12.png',
    'InfoVisJ.339.10.png',
    'InfoVisJ.924.5.png',
    'InfoVisC.57.4.png',
    'InfoVisJ.2072.16.png',
    'visMost786.png',
    'VASTJ.200.8.png',
    'treasuryG07_2.png',
    'InfoVisJ.1247.14.png',
    'InfoVisJ.2556.7(2).png',
    'vis393.png',
    'InfoVisC.65.5(2).png',
    'InfoVisJ.464.11.png',
]

df = pd.read_csv(MAIN, encoding='cp1252')
df_sel = df[df['imageName'].isin(IMAGES)].copy()

# Reorder to match the image list order
df_sel['_order'] = df_sel['imageName'].map({k: i for i, k in enumerate(IMAGES)})
df_sel = df_sel.sort_values('_order').drop(columns='_order').reset_index(drop=True)

out = pd.DataFrame({
    'imageName':                    df_sel['imageName'],
    'imageURL':                     df_sel['imageURL'],
    'VisType':                      df_sel['VisType'],
    'NormalizedVC':                 df_sel['NormalizedVC'],
    'originalPhrases\n(LLM input)': df_sel['CuratePhrasesMore/LessComplex'],
    'Topics':                       df_sel['UniqueTopics'],
    'SubTopics\n(LLM input)':       df_sel['UniqueSubTopics'],
    'objectWords\n(LLM input)':     df_sel['objectWords'],
    'actionWords':                  df_sel['actionWords'],
    'attributes':                   df_sel['oar_B_synset_attributes'],
    'relationships':                df_sel['oar_B_synset_relationships'],
})

out.to_csv(OUT, index=False)
print(f'Saved {len(out)} rows to {OUT.relative_to(ROOT)}')
for _, row in out.iterrows():
    print(f"  {row['imageName']:35s}  {row['VisType']:12s}  VC={row['NormalizedVC']:.2f}")
