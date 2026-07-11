"""
Mirage Score Experiment — Text-Only VC Scoring (No Image)
==========================================================
Replicates the V0+TW-dyn scoring prompt but sends TEXT instead of an image:

  --text-mode comment     : send the raw user comment(s) for the image
  --text-mode explanation : send the prior LLM explanation (from vc_explanations.csv)

Supports both Anthropic (claude-sonnet-4-6, etc.) and OpenAI (gpt-5.4, etc.) models.

Usage:
    # Comment mode — Claude
    python _vc_score_api_mirage_text.py \
        --input-csv ../../results/vc_api_63_v0_tw_dyn/vc_scores.csv \
        --text-mode comment \
        --model claude-sonnet-4-6 \
        --outdir ../../results/vc_api_63_mirage_comment_claude \
        --concurrency 5

    # Explanation mode — Claude
    python _vc_score_api_mirage_text.py \
        --input-csv ../../results/vc_api_63_v0_tw_dyn/vc_scores.csv \
        --text-mode explanation \
        --model claude-sonnet-4-6 \
        --outdir ../../results/vc_api_63_mirage_explanation_claude \
        --concurrency 5

    # Comment mode — GPT
    python _vc_score_api_mirage_text.py \
        --input-csv ../../results/vc_api_63_v0_tw_dyn/vc_scores.csv \
        --text-mode comment \
        --model gpt-5.4 \
        --outdir ../../results/vc_api_63_mirage_comment_gpt \
        --concurrency 5

    # Explanation mode — GPT
    python _vc_score_api_mirage_text.py \
        --input-csv ../../results/vc_api_63_v0_tw_dyn/vc_scores.csv \
        --text-mode explanation \
        --model gpt-5.4 \
        --outdir ../../results/vc_api_63_mirage_explanation_gpt \
        --concurrency 5
"""

import os, sys, json, time, argparse, csv, asyncio, threading
from pathlib import Path
import anthropic
import openai
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

MODEL           = 'claude-sonnet-4-6'
MAX_TOKENS      = 800
SLEEP_BETWEEN   = 0.5

# Default data sources
COMMENTS_CSV    = Path(__file__).parent.parent.parent / 'comment_process' / 'ResultsStepByStep - 4.0.imageDataCompiled.csv'
EXPLANATIONS_CSV = Path(__file__).parent.parent.parent / 'results' / 'vc_api_63_v0_tw_dyn' / 'vc_explanations.csv'
DEFAULT_INPUT_CSV = Path(__file__).parent.parent.parent / 'results' / 'vc_api_63_v0_tw_dyn' / 'vc_scores.csv'

# ── System Prompt — same as V0+TW-dyn, but input is text, not image ──────────
# A single neutral prompt is used for both comment and explanation modes.
# Hiding the source identity (comment vs AI explanation) avoids framing
# confounds: the only variable between conditions is the text content itself.
SYSTEM_PROMPT_TEXT = """You are a visual complexity (VC) scoring expert for data visualizations.

**Visual Complexity** measures the amount of detail, intricacy, and perceptual/cognitive demand of a visualization image. It is scored on a 0–1 scale where 0 = minimal complexity and 1 = maximum complexity.

You will receive a text description of a visualization. Score its visual complexity based solely on the information in this text — you do NOT have access to the actual image.

## Topics to Consider When Scoring

When assessing visual complexity, consider the following 7 topics that commonly drive complexity perception. You do not need to score each topic individually — use them as mental checkpoints to arrive at a holistic vc_score.

1. **Data Density / Image Clutter** — The perceived amount, richness, or depth of data content. Considers information volume, element quantity, and visual clutter/overlap.
2. **Visual Encoding Clarity** — The variety, type, and complexity of graphical forms (shapes, lines, marks) and how spatial layout, scale, and encoding interpretability contribute to complexity.
3. **Semantics / Text Legibility** — The quantity and density of text elements (titles, axis labels, legends, captions, annotations, in-chart labels).
4. **Schema** — Whether specialized domain knowledge is needed, including dimensionality (2D/3D), structural complexity, and abstraction level.
5. **Color, Symbol, and Texture Details** — Range, variety, and arrangement of colors, plus use of symbols, textures, and non-color graphical markers.
6. **Aesthetics Uncertainty** — How visually cluttered, dense, or disordered the layout appears. Higher = more cluttered/overwhelming. A clean minimal layout scores low; a crowded layout with overlapping elements scores high.
7. **Immediacy / Cognitive Load** — Overall ease or difficulty of interpreting the visualization. Considers interpretive difficulty, semantic clarity, and processing time/effort.

## Weighting of Topics for the Overall vc_score
Before arriving at the final vc_score, examine which of the 7 topics are most salient based on the provided text. Assign your own relative importance to each topic based on what the text reveals — no fixed priority is imposed. Let the described characteristics guide which topics matter most for this particular case.

## Output Format
Return ONLY valid JSON (no markdown fences, no explanation outside JSON):
{
  "vc_score": <float 0-1, weighted holistic judgment>,
  "explanation": "<2-3 sentence justification referencing the topics that most influenced the score>"
}"""


# ── Helpers ─────────────────────────────────────────────────────────────────

def is_openai_model(model: str) -> bool:
    return model.startswith('gpt') or model.startswith('o1') or model.startswith('o3') or model.startswith('o4')


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


def load_text_inputs(text_mode: str, image_names: list) -> dict:
    """Return dict mapping imageName -> text string for the given mode."""
    import pandas as pd

    if text_mode == 'comment':
        df = pd.read_csv(COMMENTS_CSV)
        # Only process the requested images to avoid noise from the full dataset
        df = df[df['imageName'].isin(image_names)]
        comment_map = {}
        for name, group in df.groupby('imageName'):
            parts = [c.strip() for c in group['rawUserComments'].dropna().tolist() if str(c).strip()]
            if parts:
                comment_map[name] = '\n\n'.join(f'Comment {i+1}: {c}' for i, c in enumerate(parts))
            else:
                # Fallback: use CuratePhrasesMore/LessComplex when rawUserComments is empty
                fallback = [c.strip() for c in group['CuratePhrasesMore/LessComplex'].dropna().tolist() if str(c).strip()]
                if fallback:
                    comment_map[name] = f'Description: {"; ".join(fallback)}'
                    print(f'  Fallback used for {name}: {comment_map[name]}')
        return comment_map

    elif text_mode == 'explanation':
        df = pd.read_csv(EXPLANATIONS_CSV)
        return dict(zip(df['filename'], df['explanation']))

    else:
        raise ValueError(f"Unknown text_mode: {text_mode!r}. Use 'comment' or 'explanation'.")


_csv_lock = threading.Lock()


# ── Anthropic async scoring ──────────────────────────────────────────────────

async def _score_one_anthropic(aclient, sem, idx, total, img_name, text_input,
                                system_prompt, scores_csv, expl_csv):
    async with sem:
        if not text_input or not str(text_input).strip():
            print(f'[{idx}/{total}] {img_name} ... SKIP (no text input)')
            return (img_name, None)

        user_text = f"Score the visual complexity of this visualization based on the following text:\n\n{text_input}"
        messages = [{'role': 'user', 'content': user_text}]  # source identity hidden

        try:
            response = await aclient.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=0,
                system=[{"type": "text", "text": system_prompt}],
                messages=messages,
            )
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


async def _run_concurrent_anthropic(api_key, to_process, text_map, system_prompt,
                                     scores_csv, expl_csv, concurrency):
    aclient = anthropic.AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    total = len(to_process)
    print(f'Concurrent mode: {concurrency} workers\n')

    tasks = [
        _score_one_anthropic(aclient, sem, i, total, name, text_map.get(name, ''),
                              system_prompt, scores_csv, expl_csv)
        for i, name in enumerate(to_process, 1)
    ]
    results = await asyncio.gather(*tasks)
    ok = sum(1 for _, vc in results if vc is not None)
    failed = [name for name, vc in results if vc is None]
    return ok, failed


# ── OpenAI async scoring ─────────────────────────────────────────────────────

async def _score_one_openai(aclient, sem, idx, total, img_name, text_input,
                             system_prompt, scores_csv, expl_csv):
    async with sem:
        if not text_input or not str(text_input).strip():
            print(f'[{idx}/{total}] {img_name} ... SKIP (no text input)')
            return (img_name, None)

        user_text = f"Score the visual complexity of this visualization based on the following text:\n\n{text_input}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_text},
        ]

        try:
            response = await aclient.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_completion_tokens=MAX_TOKENS,
            )
            raw = response.choices[0].message.content
            result = parse_response(raw)

            with _csv_lock:
                append_scores_row(scores_csv, img_name, result)
                append_explanations_row(expl_csv, img_name, result)

            vc = result.get('vc_score', '?')
            print(f'[{idx}/{total}] {img_name} ... OK  (vc_score={vc})')
            return (img_name, vc)

        except openai.RateLimitError:
            print(f'[{idx}/{total}] {img_name} ... RATE LIMITED')
            await asyncio.sleep(60)
            return (img_name, None)
        except Exception as e:
            print(f'[{idx}/{total}] {img_name} ... ERROR: {e}')
            return (img_name, None)


async def _run_concurrent_openai(api_key, to_process, text_map, system_prompt,
                                  scores_csv, expl_csv, concurrency):
    aclient = openai.AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    total = len(to_process)
    print(f'Concurrent mode: {concurrency} workers\n')

    tasks = [
        _score_one_openai(aclient, sem, i, total, name, text_map.get(name, ''),
                          system_prompt, scores_csv, expl_csv)
        for i, name in enumerate(to_process, 1)
    ]
    results = await asyncio.gather(*tasks)
    ok = sum(1 for _, vc in results if vc is not None)
    failed = [name for name, vc in results if vc is None]
    return ok, failed


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Mirage Text-Only VC Scoring (comment or explanation)')
    parser.add_argument('--text-mode', choices=['comment', 'explanation'], required=True,
                        help='Input text source: user comments or prior LLM explanations')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (e.g. claude-sonnet-4-6, gpt-5.4)')
    parser.add_argument('--input-csv', type=str, default=None,
                        help='CSV with a "filename" or "imageName" column listing images to process. '
                             'Defaults to vc_api_63_v0_tw_dyn/vc_scores.csv')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Output directory for vc_scores.csv and vc_explanations.csv')
    parser.add_argument('--concurrency', type=int, default=5)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    global MODEL
    if args.model:
        MODEL = args.model

    use_openai = is_openai_model(MODEL)
    system_prompt = SYSTEM_PROMPT_TEXT

    if use_openai:
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print('ERROR: OPENAI_API_KEY not set. Check your .env file.')
            sys.exit(1)
    else:
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            print('ERROR: ANTHROPIC_API_KEY not set. Check your .env file.')
            sys.exit(1)

    import pandas as pd
    input_csv_path = Path(args.input_csv) if args.input_csv else DEFAULT_INPUT_CSV
    input_df = pd.read_csv(input_csv_path)
    # Normalise column name: accept both 'filename' and 'imageName'
    if 'imageName' in input_df.columns:
        image_names = input_df['imageName'].drop_duplicates().tolist()
    elif 'filename' in input_df.columns:
        image_names = input_df['filename'].drop_duplicates().tolist()
    else:
        print('ERROR: Input CSV must have a "filename" or "imageName" column.')
        sys.exit(1)

    if args.limit:
        image_names = image_names[:args.limit]

    # Default output directory
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        provider = 'gpt' if use_openai else 'claude'
        outdir = Path(__file__).parent.parent.parent / 'results' / f'vc_api_63_mirage_{args.text_mode}_{provider}'
    outdir.mkdir(parents=True, exist_ok=True)
    scores_csv = outdir / 'vc_scores.csv'
    expl_csv   = outdir / 'vc_explanations.csv'

    done = set() if args.overwrite else load_existing_scores(scores_csv)
    to_process = [n for n in image_names if n not in done]
    skip_count = len(image_names) - len(to_process)

    print(f'=== Mirage Text-Only VC Scoring ===')
    print(f'Text mode:      {args.text_mode}')
    print(f'Model:          {MODEL}')
    print(f'Provider:       {"OpenAI" if use_openai else "Anthropic"}')
    print(f'Output dir:     {outdir.resolve()}')
    print(f'Total images:   {len(image_names)}')
    print(f'Already done:   {skip_count}')
    print(f'To process:     {len(to_process)}')
    print()

    if not to_process:
        print('Nothing to process.')
        return

    print(f'Loading text inputs ({args.text_mode}) ...')
    text_map = load_text_inputs(args.text_mode, image_names)
    available = sum(1 for n in to_process if text_map.get(n, '').strip())
    print(f'  Text inputs found: {available}/{len(to_process)}\n')

    concurrency = max(1, args.concurrency)

    if use_openai:
        ok, failed = asyncio.run(_run_concurrent_openai(
            api_key, to_process, text_map, system_prompt,
            scores_csv, expl_csv, concurrency))
    else:
        ok, failed = asyncio.run(_run_concurrent_anthropic(
            api_key, to_process, text_map, system_prompt,
            scores_csv, expl_csv, concurrency))

    print(f'\n{"="*50}')
    print(f'Done: {ok} scored, {len(failed)} failed')
    if failed:
        print('Failed images:')
        for f in failed:
            print(f'  {f}')


if __name__ == '__main__':
    main()
