"""
Claude API Visual Complexity (VC) Scoring — V0 Pure Zero-Shot Baseline
======================================================================
Minimal prompt: only the definition of Visual Complexity.
No dimension descriptions, no calibration guidance, no anchors.
The model returns a single vc_score plus a brief explanation.

Usage:
    python _vc_score_api_v0.py --input-csv ../Claude_vc_prediction/gt_all_46.csv --outdir ../vc_api_46gt_v0
    python _vc_score_api_v0.py --input-csv ../Claude_vc_prediction/gt_all_46.csv --outdir ../vc_api_46gt_v0 --concurrency 5
"""

import os, sys, json, time, argparse, base64, csv, asyncio, threading
from pathlib import Path
import anthropic
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

MODEL          = 'claude-sonnet-4-6'
MAX_TOKENS     = 800
THINKING_BUDGET = 4000     # extended thinking budget (used with --thinking flag)
USE_THINKING   = False     # set at runtime by --thinking flag
SLEEP_BETWEEN  = 0.5

MAPPING_CSV    = Path(__file__).parent.parent / 'phrase_reduction_v2' / 'image_phrase_word_mapping.csv'
DEFAULT_OUTDIR = Path(__file__).parent.parent / 'vc_api_46gt_v0'

# ── V0 System Prompt — Definition Only ─────────────────────────────────────
SYSTEM_PROMPT = """You are a visual complexity (VC) scoring expert for data visualizations.

**Visual Complexity** measures the amount of detail, intricacy, and perceptual/cognitive demand of a visualization image. It is scored on a 0–1 scale where 0 = minimal complexity and 1 = maximum complexity.

You will receive a single visualization image with NO accompanying text or comments. Score it purely from what you see.

## Output Format
Return ONLY valid JSON (no markdown fences, no explanation outside JSON):
{
  "vc_score": <float 0-1>,
  "explanation": "<2-3 sentence justification>"
}"""


# ── Helpers ─────────────────────────────────────────────────────────────────

def image_url_to_base64(url: str) -> str:
    import urllib.request
    with urllib.request.urlopen(url, timeout=30) as resp:
        return base64.standard_b64encode(resp.read()).decode('utf-8')


def get_media_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
            'gif': 'image/gif', 'webp': 'image/webp'}.get(ext.lstrip('.'), 'image/png')


def _load_image(img_name: str, url: str = None) -> str | None:
    target = url or f'https://raw.githubusercontent.com/c109363/ExperimentImage/main/AllDataResize/{img_name}'
    try:
        return image_url_to_base64(target)
    except Exception as e:
        print(f'  WARNING: Failed to fetch {img_name}: {e}')
        return None


def parse_response(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def load_existing_scores(scores_csv: Path) -> set:
    if not scores_csv.exists():
        return set()
    done = set()
    with open(scores_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row['filename'])
    return done


def append_scores_row(scores_csv: Path, filename: str, result: dict):
    write_header = not scores_csv.exists()
    with open(scores_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['filename', 'vc_score'])
        writer.writerow([filename, result.get('vc_score', '')])


def append_explanations_row(expl_csv: Path, filename: str, result: dict):
    write_header = not expl_csv.exists()
    with open(expl_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['filename', 'explanation'])
        writer.writerow([filename, result.get('explanation', '')])


_csv_lock = threading.Lock()


# ── Async scoring ──────────────────────────────────────────────────────────

async def _score_one_async(aclient, sem, idx, total, img_name, img_url,
                           scores_csv, expl_csv):
    async with sem:
        b64 = _load_image(img_name, url=img_url)
        if b64 is None:
            print(f'[{idx}/{total}] {img_name} ... SKIP (image not found)')
            return (img_name, None)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": get_media_type(img_name), "data": b64}
                    },
                    {"type": "text", "text": "Score the visual complexity of this visualization image."}
                ]
            }
        ]

        try:
            api_kwargs = dict(model=MODEL, messages=messages,
                              max_tokens=MAX_TOKENS + THINKING_BUDGET if USE_THINKING else MAX_TOKENS,
                              system=[{"type": "text", "text": SYSTEM_PROMPT}])
            if USE_THINKING:
                api_kwargs['thinking'] = {"type": "adaptive"}
            else:
                api_kwargs['temperature'] = 0
            response = await aclient.messages.create(**api_kwargs)
            raw = next(b.text for b in response.content if b.type == 'text')
            result = parse_response(raw)

            with _csv_lock:
                append_scores_row(scores_csv, img_name, result)
                append_explanations_row(expl_csv, img_name, result)

            vc = result.get('vc_score', '?')
            print(f'[{idx}/{total}] {img_name} ... OK  (vc_score={vc})')
            return (img_name, vc)

        except anthropic.RateLimitError:
            print(f'[{idx}/{total}] {img_name} ... RATE LIMITED')
            await asyncio.sleep(60)
            return (img_name, None)
        except Exception as e:
            print(f'[{idx}/{total}] {img_name} ... ERROR: {e}')
            return (img_name, None)


async def _run_concurrent(api_key, to_process, scores_csv, expl_csv, concurrency):
    aclient = anthropic.AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    total = len(to_process)
    print(f'Concurrent mode: {concurrency} workers\n')

    tasks = [
        _score_one_async(aclient, sem, i, total, row['imageName'], row['imageURL'],
                         scores_csv, expl_csv)
        for i, row in enumerate(to_process, 1)
    ]

    results = await asyncio.gather(*tasks)
    ok = sum(1 for _, vc in results if vc is not None)
    failed = [name for name, vc in results if vc is None]
    return ok, failed


def _run_sequential(client, to_process, scores_csv, expl_csv):
    ok, failed = 0, []
    for i, row in enumerate(to_process, 1):
        img_name = row['imageName']
        img_url  = row['imageURL']
        print(f'[{i}/{len(to_process)}] {img_name} ... ', end='', flush=True)
        try:
            b64 = _load_image(img_name, url=img_url)
            if b64 is None:
                print('SKIP (image not found)')
                failed.append(img_name)
                continue

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": get_media_type(img_name), "data": b64}
                        },
                        {"type": "text", "text": "Score the visual complexity of this visualization image."}
                    ]
                }
            ]

            api_kwargs = dict(model=MODEL, messages=messages,
                              max_tokens=MAX_TOKENS + THINKING_BUDGET if USE_THINKING else MAX_TOKENS,
                              system=[{"type": "text", "text": SYSTEM_PROMPT}])
            if USE_THINKING:
                api_kwargs['thinking'] = {"type": "adaptive"}
            else:
                api_kwargs['temperature'] = 0
            response = client.messages.create(**api_kwargs)
            raw = next(b.text for b in response.content if b.type == 'text')
            result = parse_response(raw)

            append_scores_row(scores_csv, img_name, result)
            append_explanations_row(expl_csv, img_name, result)

            print(f'OK  (vc_score={result.get("vc_score", "?")})')
            ok += 1
        except json.JSONDecodeError as e:
            print(f'JSON PARSE ERROR: {e}')
            failed.append(img_name)
        except anthropic.RateLimitError:
            print('RATE LIMITED — waiting 60s ...')
            time.sleep(60)
            failed.append(img_name)
        except Exception as e:
            print(f'ERROR: {e}')
            failed.append(img_name)
        if i < len(to_process):
            time.sleep(SLEEP_BETWEEN)
    return ok, failed


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='V0 Pure Zero-Shot VC Scoring via Claude API')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--input-csv', type=str, default=None,
                        help='CSV with imageName column')
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--thinking', action='store_true',
                        help='Enable extended thinking (budget_tokens=16000, temperature=1)')
    parser.add_argument('--model', type=str, default=None,
                        help='Override model (e.g. claude-opus-4-6 or claude-sonnet-4-6)')
    args = parser.parse_args()
    global USE_THINKING, MODEL
    USE_THINKING = args.thinking
    if args.model:
        MODEL = args.model

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print('ERROR: ANTHROPIC_API_KEY not set. Check your .env file.')
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # ── Load image list ─────────────────────────────────────────────────
    import pandas as pd
    if args.input_csv:
        input_df = pd.read_csv(args.input_csv)
        if 'imageURL' not in input_df.columns:
            mapping_df = pd.read_csv(MAPPING_CSV)
            url_map = dict(zip(mapping_df['imageName'], mapping_df['imageURL']))
            input_df['imageURL'] = input_df['imageName'].map(url_map).fillna('')
        all_images = input_df[['imageName', 'imageURL']].drop_duplicates('imageName').to_dict('records')
    else:
        mapping_df = pd.read_csv(MAPPING_CSV)
        all_images = mapping_df[['imageName', 'imageURL']].drop_duplicates('imageName').to_dict('records')

    if args.limit:
        all_images = all_images[:args.limit]

    # ── Output paths ────────────────────────────────────────────────────
    outdir = Path(args.outdir) if args.outdir else DEFAULT_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    scores_csv = outdir / 'vc_scores.csv'
    expl_csv   = outdir / 'vc_explanations.csv'

    done = set() if args.overwrite else load_existing_scores(scores_csv)
    to_process = [r for r in all_images if r['imageName'] not in done]
    skip_count = len(all_images) - len(to_process)

    print(f'=== V0 Pure Zero-Shot Baseline ===')
    print(f'Model:          {MODEL}')
    print(f'Prompt:         Definition only (no dimensions, no anchors)')
    print(f'Output dir:     {outdir.resolve()}')
    print(f'Total images:   {len(all_images)}')
    print(f'Already done:   {skip_count}')
    print(f'To process:     {len(to_process)}')
    print()

    if not to_process:
        print('Nothing to process.')
        return

    concurrency = max(1, args.concurrency)
    if concurrency == 1:
        ok, failed = _run_sequential(client, to_process, scores_csv, expl_csv)
    else:
        ok, failed = asyncio.run(
            _run_concurrent(api_key, to_process, scores_csv, expl_csv, concurrency)
        )

    print(f'\n{"="*50}')
    print(f'Done: {ok} scored, {len(failed)} failed')
    if failed:
        print(f'Failed images:')
        for f in failed:
            print(f'  {f}')
    print(f'Scores:       {scores_csv.resolve()}')
    print(f'Explanations: {expl_csv.resolve()}')


if __name__ == '__main__':
    main()
