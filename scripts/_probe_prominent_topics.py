"""
Option-1 probe: "prominent topics" (variable cardinality) vs top-3 on 9 images.
Calls the API with a modified V0+T prompt that asks for all distinctly prominent
topics rather than exactly top-3. Results are shown side-by-side with:
  - Human GT topics
  - Top-3 prediction (from existing vc_api_510_topicsel_v0_t run)
  - Prominent-only prediction (new)
"""
import os, json, base64, asyncio
from pathlib import Path
import urllib.request
import anthropic
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

ROOT = Path(__file__).parent.parent

IMAGES = [
    ('economist_daily_chart_493.png', 'Area',         'Data Density / Image Clutter; Schema; Color, Symbol, and Texture Details'),
    ('wsj28.png',                     'Bar',           'Semantics / Text Legibility; Color, Symbol, and Texture Details'),
    ('v483_n7391_8_f5.png',           'Cont.-ColorPatn','Data Density / Image Clutter; Visual Encoding Clarity; Semantics / Text Legibility; Schema; Color, Symbol, and Texture Details; Aesthetics Uncertainty; Immediacy / Cognitive Load'),
    ('SciVisJ.980.7(2).png',          'Glyph',         'Immediacy / Cognitive Load'),
    ('VASTC.115.3.png',               'Grid',          'Immediacy / Cognitive Load'),
    ('whoB09_1.png',                  'Line',          'Data Density / Image Clutter; Semantics / Text Legibility; Color, Symbol, and Texture Details; Immediacy / Cognitive Load'),
    ('np_11.png',                     'Node-link',     'Data Density / Image Clutter; Visual Encoding Clarity; Semantics / Text Legibility; Schema; Color, Symbol, and Texture Details'),
    ('wsj533.png',                    'Point',         'Semantics / Text Legibility; Aesthetics Uncertainty'),
    ('InfoVisJ.2412.17(2).png',       'Text',          'Immediacy / Cognitive Load'),
]

BASE_URL = 'https://raw.githubusercontent.com/c109363/ExperimentImage/main/AllDataResize/'

TOPIC_LABEL_TO_KEY = {
    'Data Density / Image Clutter':      'data_density',
    'Visual Encoding Clarity':           'visual_encoding',
    'Semantics / Text Legibility':       'text_annotation',
    'Schema':                            'domain_schema',
    'Color, Symbol, and Texture Details':'color_symbol',
    'Aesthetics Uncertainty':            'aesthetic_order',
    'Immediacy / Cognitive Load':        'cognitive_load',
}
KEY_TO_SHORT = {
    'data_density':   'DataDensity',
    'visual_encoding':'VisEncoding',
    'text_annotation':'TextAnnot',
    'domain_schema':  'Schema',
    'color_symbol':   'ColorSymbol',
    'aesthetic_order':'Aesthetics',
    'cognitive_load': 'CogLoad',
}

# ── Option-1 system prompt ───────────────────────────────────────────────────
SYSTEM_PROMINENT = """You are a visual complexity (VC) scoring expert for data visualizations.

**Visual Complexity** measures the amount of detail, intricacy, and perceptual/cognitive demand of a visualization image. It is scored on a 0–1 scale where 0 = minimal complexity and 1 = maximum complexity.

You will receive a single visualization image with NO accompanying text or comments. Score it purely from what you see.

## Topics to Consider When Scoring

When assessing visual complexity, consider the following 7 topics that commonly drive complexity perception:

1. **Data Density / Image Clutter** (key: data_density)
2. **Visual Encoding Clarity** (key: visual_encoding)
3. **Semantics / Text Legibility** (key: text_annotation)
4. **Schema** (key: domain_schema)
5. **Color, Symbol, and Texture Details** (key: color_symbol)
6. **Aesthetics Uncertainty** (key: aesthetic_order)
7. **Immediacy / Cognitive Load** (key: cognitive_load)

## Task
Identify which of the 7 topics are **distinctly prominent** in THIS image — meaning they make a clear, substantial contribution to the visual complexity you perceive. Do not list topics that are merely present but unremarkable. The number of prominent topics can range from 1 to 7 depending on the image.

## Output Format
Return ONLY valid JSON:
{
  "vc_score": <float 0-1>,
  "prominent_topics": ["key1", "key2", ...],
  "explanation": "<1-2 sentences>"
}"""


def fetch_b64(img_name: str) -> str:
    url = BASE_URL + img_name
    with urllib.request.urlopen(url, timeout=30) as r:
        return base64.standard_b64encode(r.read()).decode()


async def score_one(aclient, img_name: str) -> dict:
    b64 = await asyncio.to_thread(fetch_b64, img_name)
    ext = img_name.rsplit('.', 1)[-1].lower()
    mt = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png'}.get(ext, 'image/png')
    response = await aclient.messages.create(
        model='claude-opus-4-6',
        max_tokens=400,
        temperature=0,
        system=[{"type": "text", "text": SYSTEM_PROMINENT}],
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mt, "data": b64}},
            {"type": "text", "text": "Score the visual complexity and identify the distinctly prominent topics."}
        ]}]
    )
    raw = next(b.text for b in response.content if b.type == 'text').strip()
    if raw.startswith('```'):
        raw = raw.split('```')[1]
        if raw.startswith('json'):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


async def main():
    api_key = os.environ['ANTHROPIC_API_KEY']
    aclient = anthropic.AsyncAnthropic(api_key=api_key)

    # Load existing top-3 predictions for both variants
    import pandas as pd
    vt_csv   = ROOT / 'results' / 'vc_api_510_topicsel_v0_t'  / 'vc_scores.csv'
    vtw_csv  = ROOT / 'results' / 'vc_api_510_topicsel_v0_tw' / 'vc_scores.csv'
    vt_df  = pd.read_csv(vt_csv).set_index('filename')
    vtw_df = pd.read_csv(vtw_csv).set_index('filename')

    print(f'{"Image":<32} {"VisType":<16} {"GT topics":<45} {"V0+T top3":<30} {"V0+TW top3":<30} {"Prominent":<40} F1(T) F1(TW) F1(P)')
    print('-' * 210)

    tasks = [(img, vt, gt) for img, vt, gt in IMAGES]
    results = await asyncio.gather(*[score_one(aclient, img) for img, _, _ in tasks])

    t_f1s, tw_f1s, p_f1s = [], [], []
    for (img, vt, gt_raw), result in zip(tasks, results):
        gt_keys = [TOPIC_LABEL_TO_KEY[t.strip()] for t in gt_raw.split(';') if t.strip() in TOPIC_LABEL_TO_KEY]
        gt_short = ', '.join(KEY_TO_SHORT[k] for k in gt_keys)

        t_raw  = vt_df.loc[img,  'top3_topics'] if img in vt_df.index  else ''
        tw_raw = vtw_df.loc[img, 'top3_topics'] if img in vtw_df.index else ''
        t_keys  = [k for k in str(t_raw).split(';')  if k]
        tw_keys = [k for k in str(tw_raw).split(';') if k]
        t_short  = ', '.join(KEY_TO_SHORT.get(k, k) for k in t_keys)
        tw_short = ', '.join(KEY_TO_SHORT.get(k, k) for k in tw_keys)

        prom_keys = result.get('prominent_topics', [])
        prom_short = ', '.join(KEY_TO_SHORT.get(k, k) for k in prom_keys)

        gt_set = set(gt_keys)
        t_f1  = _f1(set(t_keys),  gt_set)
        tw_f1 = _f1(set(tw_keys), gt_set)
        p_f1  = _f1(set(prom_keys), gt_set)
        t_f1s.append(t_f1); tw_f1s.append(tw_f1); p_f1s.append(p_f1)

        print(f'{img:<32} {vt:<16} {gt_short:<45} {t_short:<30} {tw_short:<30} {prom_short:<40} {t_f1:.2f}  {tw_f1:.2f}   {p_f1:.2f}')
        print(f'  explanation: {result.get("explanation","")[:120]}')
        print()

    print(f'\nMean F1:  V0+T={sum(t_f1s)/len(t_f1s):.3f}  V0+TW={sum(tw_f1s)/len(tw_f1s):.3f}  Prominent={sum(p_f1s)/len(p_f1s):.3f}')


def _f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    p, r = tp / len(pred), tp / len(gold)
    return 2*p*r/(p+r) if (p+r) else 0.0


if __name__ == '__main__':
    asyncio.run(main())
