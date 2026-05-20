"""
Run the Option-1 "prominent topics" probe on the full 510-image set,
then compare V0+T top-3, V0+TW top-3, and Prominent side by side.

Results are saved to: results/probe_prominent_510/prominent_results.csv
Run summary printed to console.

Usage:
    python _probe_prominent_510.py
"""
import os, json, base64, asyncio, urllib.parse
from pathlib import Path
import urllib.request
import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

ROOT  = Path(__file__).parent.parent
MODEL = 'claude-sonnet-4-5'  # Sonnet variant; Opus results in probe_prominent_510_opus_4_6/

# Output directory is derived from model name so results never overwrite each other
_model_tag = MODEL.replace('claude-', '').replace('.', '_')
OUT_DIR = ROOT / 'results' / f'probe_prominent_510_{_model_tag}'
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL    = 'https://raw.githubusercontent.com/c109363/ExperimentImage/main/AllDataResize/'
CONCURRENCY = 5

COMPILED   = ROOT / 'phrase_reduction_v2' / 'image_compiled_phrases.csv'
INPUT_510  = ROOT / 'results' / 'vc_api_510_v0_tw_input.csv'
VT_CSV     = ROOT / 'results' / 'vc_api_510_topicsel_v0_t'  / 'vc_scores.csv'
VTW_CSV    = ROOT / 'results' / 'vc_api_510_topicsel_v0_tw' / 'vc_scores.csv'

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
DIMS = list(KEY_TO_SHORT.keys())

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


# ── GT ────────────────────────────────────────────────────────────────────────

def build_gt_510() -> pd.DataFrame:
    compiled = pd.read_csv(COMPILED)[['imageName', 'Topics']]
    inp      = pd.read_csv(INPUT_510)[['imageName', 'VisType']]
    merged   = inp.merge(compiled, on='imageName')

    rows = []
    for _, r in merged.iterrows():
        dims = []
        for t in str(r['Topics']).split(';'):
            t = t.strip()
            if t in TOPIC_LABEL_TO_KEY:
                d = TOPIC_LABEL_TO_KEY[t]
                if d not in dims:
                    dims.append(d)
        rows.append({
            'filename': r['imageName'],
            'vistype':  r['VisType'],
            'gt_keys':  ';'.join(dims),
        })
    return pd.DataFrame(rows)


# ── API ───────────────────────────────────────────────────────────────────────

def fetch_b64(img_name: str) -> str:
    url = BASE_URL + urllib.parse.quote(img_name)
    with urllib.request.urlopen(url, timeout=30) as resp:
        return base64.b64encode(resp.read()).decode()

sem     = None  # set in main
_done   = 0
_total  = 0

async def score_one(aclient, img_name: str) -> dict:
    global _done
    async with sem:
        b64 = await asyncio.to_thread(fetch_b64, img_name)
        ext = img_name.rsplit('.', 1)[-1].lower()
        mt  = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png'}.get(ext, 'image/png')
        response = await aclient.messages.create(
            model=MODEL,
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
        result = json.loads(raw)
        _done += 1
        if _done % 10 == 0 or _done == _total:
            print(f'  {_done}/{_total} done', flush=True)
        return result


# ── Metrics ───────────────────────────────────────────────────────────────────

def _f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r) if (p + r) else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    global sem
    sem = asyncio.Semaphore(CONCURRENCY)

    api_key  = os.environ['ANTHROPIC_API_KEY']
    aclient  = anthropic.AsyncAnthropic(api_key=api_key)

    print(f'Model: {MODEL}  →  {OUT_DIR.name}')
    gt_df    = build_gt_510()
    print(f'GT built: {len(gt_df)} images')

    vt_df    = pd.read_csv(VT_CSV).set_index('filename')
    vtw_df   = pd.read_csv(VTW_CSV).set_index('filename')

    # ── Caching ───────────────────────────────────────────────────────────────
    cache_path = OUT_DIR / 'prominent_results.csv'
    if cache_path.exists():
        cached = pd.read_csv(cache_path).set_index('filename')
        print(f'Loaded {len(cached)} cached results')
    else:
        cached = None

    images = gt_df['filename'].tolist()
    to_run = [img for img in images if cached is None or img not in cached.index]
    print(f'{len(images) - len(to_run)} cached, {len(to_run)} to run via API')

    if to_run:
        global _total
        _total = len(to_run)
        print(f'Calling API for {len(to_run)} images (concurrency={CONCURRENCY})...')
        results = await asyncio.gather(*[score_one(aclient, img) for img in to_run],
                                       return_exceptions=True)
        new_rows = []
        for img, res in zip(to_run, results):
            if isinstance(res, Exception):
                print(f'  ERROR {img}: {res}')
                new_rows.append({'filename': img, 'vc_score': None,
                                 'prominent_topics': '', 'explanation': f'ERROR: {res}'})
            else:
                new_rows.append({
                    'filename':         img,
                    'vc_score':         res.get('vc_score'),
                    'prominent_topics': ';'.join(res.get('prominent_topics', [])),
                    'explanation':      res.get('explanation', ''),
                })
        new_df = pd.DataFrame(new_rows).set_index('filename')
        full   = pd.concat([cached, new_df]) if cached is not None else new_df
        full.to_csv(cache_path)
        print(f'Saved to {cache_path}')
    else:
        full = cached

    # ── Per-image analysis ────────────────────────────────────────────────────
    rows_out = []
    t_f1s, tw_f1s, p_f1s = [], [], []

    for _, gr in gt_df.iterrows():
        img     = gr['filename']
        gt_set  = set(gr['gt_keys'].split(';')) if gr['gt_keys'] else set()

        t_raw   = vt_df.loc[img,  'top3_topics'] if img in vt_df.index  else ''
        tw_raw  = vtw_df.loc[img, 'top3_topics'] if img in vtw_df.index else ''
        t_keys  = set(k for k in str(t_raw).split(';')  if k)
        tw_keys = set(k for k in str(tw_raw).split(';') if k)

        prom_raw  = full.loc[img, 'prominent_topics'] if img in full.index else ''
        prom_keys = set(k for k in str(prom_raw).split(';') if k)

        t_f1  = _f1(t_keys,    gt_set)
        tw_f1 = _f1(tw_keys,   gt_set)
        p_f1  = _f1(prom_keys, gt_set)
        t_f1s.append(t_f1); tw_f1s.append(tw_f1); p_f1s.append(p_f1)

        rows_out.append({
            'filename': img, 'vistype': gr['vistype'],
            'gt': ';'.join(sorted(gt_set)),
            'vot_top3':  ';'.join(sorted(t_keys)),
            'votw_top3': ';'.join(sorted(tw_keys)),
            'prominent': ';'.join(sorted(prom_keys)),
            'f1_vot': t_f1, 'f1_votw': tw_f1, 'f1_prom': p_f1,
        })

    results_df = pd.DataFrame(rows_out)
    results_df.to_csv(OUT_DIR / 'analysis_510.csv', index=False)
    print(f'\nSaved per-image analysis → {OUT_DIR / "analysis_510.csv"}')

    # ── Per-vis-type summary ────────────────────────────────────────────────────
    print(f'\n{"VisType":<22} {"N":>4}  {"V0+T":>6}  {"V0+TW":>6}  {"Prom":>6}  {"Δ(P-T)":>7}  {"Δ(P-TW)":>8}')
    print('-' * 70)
    vt_summary = []
    for vt, grp in results_df.groupby('vistype'):
        n   = len(grp)
        mt  = grp['f1_vot'].mean()
        mtw = grp['f1_votw'].mean()
        mp  = grp['f1_prom'].mean()
        vt_summary.append((vt, n, mt, mtw, mp))
        print(f'{vt:<22} {n:>4}  {mt:>6.3f}  {mtw:>6.3f}  {mp:>6.3f}  {mp-mt:>+7.3f}  {mp-mtw:>+8.3f}')
    print('-' * 70)
    n   = len(results_df)
    mt  = sum(t_f1s)  / n
    mtw = sum(tw_f1s) / n
    mp  = sum(p_f1s)  / n
    print(f'{"MACRO MEAN":<22} {n:>4}  {mt:>6.3f}  {mtw:>6.3f}  {mp:>6.3f}  {mp-mt:>+7.3f}  {mp-mtw:>+8.3f}')

    # ── Per-topic F1 ────────────────────────────────────────────────────────────
    print(f'\n{"Topic":<18}  {"V0+T F1":>8}  {"V0+TW F1":>9}  {"Prom F1":>8}  {"Δ(P-T)":>7}  {"Δ(P-TW)":>8}')
    print('-' * 65)
    for dim in DIMS:
        tf_, twf_, pf_ = [], [], []
        for _, row in results_df.iterrows():
            gt_set_r  = set(row['gt'].split(';'))       if row['gt']       else set()
            t_set_r   = set(row['vot_top3'].split(';')) if row['vot_top3'] else set()
            tw_set_r  = set(row['votw_top3'].split(';'))if row['votw_top3']else set()
            p_set_r   = set(row['prominent'].split(';'))if row['prominent']else set()
            if dim in gt_set_r or dim in t_set_r:
                tf_.append(_f1({dim} & t_set_r,  {dim} & gt_set_r))
            if dim in gt_set_r or dim in tw_set_r:
                twf_.append(_f1({dim} & tw_set_r, {dim} & gt_set_r))
            if dim in gt_set_r or dim in p_set_r:
                pf_.append(_f1({dim} & p_set_r,  {dim} & gt_set_r))
        tf  = sum(tf_)  / len(tf_)  if tf_  else float('nan')
        twf = sum(twf_) / len(twf_) if twf_ else float('nan')
        pf  = sum(pf_)  / len(pf_)  if pf_  else float('nan')
        print(f'{KEY_TO_SHORT[dim]:<18}  {tf:>8.3f}  {twf:>9.3f}  {pf:>8.3f}  {pf-tf:>+7.3f}  {pf-twf:>+8.3f}')

    # ── Save vis-type summary CSV ─────────────────────────────────────────────
    vt_df_out = pd.DataFrame(vt_summary, columns=['vistype', 'n', 'f1_vot', 'f1_votw', 'f1_prom'])
    vt_df_out.to_csv(OUT_DIR / 'vistype_summary_510.csv', index=False)
    print(f'\nSaved vis-type summary → {OUT_DIR / "vistype_summary_510.csv"}')


if __name__ == '__main__':
    asyncio.run(main())
