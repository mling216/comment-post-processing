"""
Run the Option-1 "prominent topics" probe on the full 63-image pilot set,
then compare V0+T top-3, V0+TW top-3, and Prominent side by side.

Results are saved to: results/probe_prominent_63/prominent_results.csv
Run summary printed to console.
"""
import os, json, base64, asyncio
from pathlib import Path
import urllib.request
import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / 'results' / 'probe_prominent_63'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANCHORS = {'VisC.503.6.png', 'InfoVisJ.619.17.png', 'InfoVisJ.1149.6(1).png'}
BASE_URL = 'https://raw.githubusercontent.com/c109363/ExperimentImage/main/AllDataResize/'
CONCURRENCY = 5

TOPIC_LABEL_TO_KEY = {
    'Data Density / Image Clutter':       'data_density',
    'Visual Encoding Clarity':            'visual_encoding',
    'Semantics / Text Legibility':        'text_annotation',
    'Schema':                             'domain_schema',
    'Color, Symbol, and Texture Details': 'color_symbol',
    'Aesthetics Uncertainty':             'aesthetic_order',
    'Immediacy / Cognitive Load':         'cognitive_load',
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

SYSTEM_PROMINENT = """You are a visual complexity (VC) scoring expert for data visualizations.

**Visual Complexity** measures the amount of detail, intricacy, and perceptual/cognitive demand of a visualization image. It is scored on a 0–1 scale where 0 = minimal complexity and 1 = maximum complexity.

You will receive a single visualization image with NO accompanying text or comments. Score it purely from what you see.

## Topics to Consider When Scoring

1. **data_density** – Number of data points, layers, or spatial density of marks
2. **visual_encoding** – Complexity of encoding channels (colour, size, shape, position combinations)
3. **text_annotation** – Amount, density, and legibility of text, labels, and legends
4. **domain_schema** – Presence of domain-specific symbols, icons, or structural scaffolding
5. **color_symbol** – Richness of the colour palette and symbol/texture variety
6. **aesthetic_order** – Visual organisation, alignment, whitespace, and gestalt clarity
7. **cognitive_load** – Overall perceptual and cognitive demand to parse the image

## Output Format

Return ONLY a JSON object (no markdown fences) with exactly these keys:
- "vc_score": float [0–1]
- "prominent_topics": array of topic keys that are **distinctly prominent** — i.e., they make a clear, substantial contribution to the complexity of this specific image. Include between 1 and 7 keys. Omit topics that are present but minor.
- "explanation": one sentence (≤30 words) describing the main complexity drivers

Example:
{"vc_score": 0.72, "prominent_topics": ["data_density", "text_annotation", "cognitive_load"], "explanation": "Dense scatter of labelled points across multiple facets requires extensive visual parsing."}"""


def fetch_b64(img_name: str) -> str:
    url = BASE_URL + urllib.parse.quote(img_name)
    with urllib.request.urlopen(url, timeout=30) as resp:
        return base64.b64encode(resp.read()).decode()

import urllib.parse

sem = None  # set in main

async def score_one(aclient, img_name: str) -> dict:
    async with sem:
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


def _f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def build_gt_63() -> pd.DataFrame:
    """Return DataFrame with columns: filename, vistype, gt_keys (list)"""
    gt66 = pd.read_csv(ROOT / 'Claude_vc_prediction' / 'gt_all_66.csv')
    compiled = pd.read_csv(ROOT / 'phrase_reduction_v2' / 'image_compiled_phrases.csv')
    comp_idx = compiled.set_index('imageName')

    rows = []
    for img in gt66['imageName']:
        if img in ANCHORS:
            continue
        if img not in comp_idx.index:
            print(f'  WARN: {img} not in compiled phrases')
            continue
        row = comp_idx.loc[img]
        topics_raw = str(row['Topics'])
        gt_keys = []
        for t in topics_raw.split(';'):
            t = t.strip()
            if t in TOPIC_LABEL_TO_KEY and TOPIC_LABEL_TO_KEY[t] not in gt_keys:
                gt_keys.append(TOPIC_LABEL_TO_KEY[t])
        rows.append({
            'filename': img,
            'vistype': row['VisType'],
            'gt_keys': gt_keys,
        })
    return pd.DataFrame(rows)


async def main():
    global sem
    sem = asyncio.Semaphore(CONCURRENCY)

    api_key = os.environ['ANTHROPIC_API_KEY']
    aclient = anthropic.AsyncAnthropic(api_key=api_key)

    gt_df = build_gt_63()
    print(f'GT built: {len(gt_df)} images')

    vt_df  = pd.read_csv(ROOT / 'results' / 'vc_api_topicsel_v0_t'  / 'vc_scores.csv').set_index('filename')
    vtw_df = pd.read_csv(ROOT / 'results' / 'vc_api_topicsel_v0_tw' / 'vc_scores.csv').set_index('filename')

    # Check for cached results
    cache_path = OUT_DIR / 'prominent_results.csv'
    if cache_path.exists():
        print(f'Loading cached prominent results from {cache_path}')
        cached = pd.read_csv(cache_path).set_index('filename')
    else:
        cached = None

    images = gt_df['filename'].tolist()

    if cached is not None:
        # only call API for images not yet cached
        to_run = [img for img in images if img not in cached.index]
        print(f'{len(images) - len(to_run)} cached, {len(to_run)} to run')
    else:
        to_run = images
        print(f'Running API for all {len(to_run)} images')

    if to_run:
        print(f'Calling API for {len(to_run)} images...')
        results = await asyncio.gather(*[score_one(aclient, img) for img in to_run])
        new_rows = []
        for img, res in zip(to_run, results):
            new_rows.append({
                'filename': img,
                'vc_score': res.get('vc_score', None),
                'prominent_topics': ';'.join(res.get('prominent_topics', [])),
                'explanation': res.get('explanation', ''),
            })
        new_df = pd.DataFrame(new_rows).set_index('filename')
        if cached is not None:
            full = pd.concat([cached, new_df])
        else:
            full = new_df
        full.to_csv(cache_path)
        print(f'Saved to {cache_path}')
    else:
        full = cached

    # ── Analysis ──────────────────────────────────────────────────────────────
    rows_out = []
    t_f1s, tw_f1s, p_f1s = [], [], []

    for _, gr in gt_df.iterrows():
        img = gr['filename']
        gt_keys = gr['gt_keys']
        gt_set = set(gt_keys)

        t_raw  = vt_df.loc[img,  'top3_topics'] if img in vt_df.index  else ''
        tw_raw = vtw_df.loc[img, 'top3_topics'] if img in vtw_df.index else ''
        t_keys  = [k for k in str(t_raw).split(';')  if k]
        tw_keys = [k for k in str(tw_raw).split(';') if k]

        prom_raw = full.loc[img, 'prominent_topics'] if img in full.index else ''
        prom_keys = [k for k in str(prom_raw).split(';') if k]

        t_f1  = _f1(set(t_keys),  gt_set)
        tw_f1 = _f1(set(tw_keys), gt_set)
        p_f1  = _f1(set(prom_keys), gt_set)
        t_f1s.append(t_f1); tw_f1s.append(tw_f1); p_f1s.append(p_f1)

        rows_out.append({
            'filename': img,
            'vistype': gr['vistype'],
            'gt': ';'.join(gt_keys),
            'vot_top3': ';'.join(t_keys),
            'votw_top3': ';'.join(tw_keys),
            'prominent': ';'.join(prom_keys),
            'f1_vot': t_f1,
            'f1_votw': tw_f1,
            'f1_prom': p_f1,
        })

    results_df = pd.DataFrame(rows_out)
    results_df.to_csv(OUT_DIR / 'analysis_63.csv', index=False)

    # ── Per-vistype summary ───────────────────────────────────────────────────
    print(f'\n{"VisType":<20} {"N":>3}  {"V0+T":>6}  {"V0+TW":>6}  {"Prom":>6}  {"Δ(P-T)":>7}  {"Δ(P-TW)":>8}')
    print('-' * 65)
    for vt, grp in results_df.groupby('vistype'):
        n = len(grp)
        mt  = grp['f1_vot'].mean()
        mtw = grp['f1_votw'].mean()
        mp  = grp['f1_prom'].mean()
        print(f'{vt:<20} {n:>3}  {mt:>6.3f}  {mtw:>6.3f}  {mp:>6.3f}  {mp-mt:>+7.3f}  {mp-mtw:>+8.3f}')

    print('-' * 65)
    n = len(results_df)
    mt  = sum(t_f1s)  / n
    mtw = sum(tw_f1s) / n
    mp  = sum(p_f1s)  / n
    print(f'{"MACRO MEAN":<20} {n:>3}  {mt:>6.3f}  {mtw:>6.3f}  {mp:>6.3f}  {mp-mt:>+7.3f}  {mp-mtw:>+8.3f}')

    # ── Per-topic precision/recall ────────────────────────────────────────────
    dims = list(KEY_TO_SHORT.keys())
    print(f'\nPer-topic macro-F1:')
    print(f'{"Topic":<18}  {"V0+T F1":>8}  {"V0+TW F1":>9}  {"Prom F1":>8}')
    print('-' * 50)
    for dim in dims:
        tf_list, twf_list, pf_list = [], [], []
        for _, row in results_df.iterrows():
            gt_set = set(row['gt'].split(';')) if row['gt'] else set()
            t_set  = set(row['vot_top3'].split(';'))  if row['vot_top3']  else set()
            tw_set = set(row['votw_top3'].split(';')) if row['votw_top3'] else set()
            p_set  = set(row['prominent'].split(';')) if row['prominent'] else set()
            # per-image binary: was dim in pred given it's in GT?
            if dim in gt_set or dim in t_set:
                tf_list.append(_f1({dim} & t_set, {dim} & gt_set))
            if dim in gt_set or dim in tw_set:
                twf_list.append(_f1({dim} & tw_set, {dim} & gt_set))
            if dim in gt_set or dim in p_set:
                pf_list.append(_f1({dim} & p_set, {dim} & gt_set))
        tf_  = sum(tf_list)  / len(tf_list)  if tf_list  else float('nan')
        twf_ = sum(twf_list) / len(twf_list) if twf_list else float('nan')
        pf_  = sum(pf_list)  / len(pf_list)  if pf_list  else float('nan')
        print(f'{KEY_TO_SHORT[dim]:<18}  {tf_:>8.3f}  {twf_:>9.3f}  {pf_:>8.3f}')


if __name__ == '__main__':
    asyncio.run(main())
