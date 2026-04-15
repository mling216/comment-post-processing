"""
Claude API Stage 1 Extraction Script
=====================================
Calls Claude API for each image in image_compiled_phrases.csv,
extracting objects, attributes, and relationships grounded in participant comments.
Results saved incrementally to vc_genome_output/llm_extractions_api.json.
Already-processed images are skipped on rerun.

Usage:
    python _extract_stage1_api.py                  # process all images
    python _extract_stage1_api.py --limit 10       # process first N images
    python _extract_stage1_api.py --overwrite      # re-process already done images
"""

import os, sys, json, time, argparse
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import anthropic

# ── Config ──────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

DATA_CSV  = Path('phrase_reduction_v2/image_compiled_phrases.csv')
SUB_CSV   = Path('phrase_reduction_v2/phrase_shortlist.csv')
OUT_FILE  = Path('vc_genome_output/llm_extractions_api.json')
MODEL     = 'claude-sonnet-4-6'
MAX_TOKENS = 1024
SLEEP_BETWEEN = 0.5   # seconds between calls (rate limit safety)

# ── Subtopics block (cached) ──────────────────────────────────────────────
def build_subtopics_text(subtopics_df):
    lines = []
    for _, r in subtopics_df.iterrows():
        lines.append(f"- {r['SubTopic']}: {r['Description']}")
    return '\n'.join(lines)

# ── System prompt (will be cached) ───────────────────────────────────────
SYSTEM_PROMPT = """You are a visual complexity annotation expert. Your task is to extract structured scene graph elements from participant comments about visualization images.

Given participant comments and curated phrases about a visualization image, extract:
1. **Objects** — visual elements explicitly or implicitly mentioned (e.g., bar, legend, color, text, axis_label, data, pattern)
2. **Attributes** — descriptive properties of those objects, with:
   - sentiment: "+" if the attribute increases perceived complexity, "-" if it decreases it
   - subtopic: the most relevant subtopic from the taxonomy (see below)
3. **Relationships** — subject-verb-object triples grounded in the comments (e.g., color --obscures--> information)
   - Each relationship needs: subj (object id), pred (verb/relation), obj (object id), sentiment (+/-), subtopic

Guidelines:
- Extract 3–7 objects, 3–8 attributes, 2–5 relationships
- Object names must be single lowercase words or snake_case (e.g., axis_label, color_code)
- Assign each object a region from: data_area, axes, legend, title, annotation, overall
- Attribute text should be snake_case, max 4 words (e.g., dense_overlapping_marks)
- Predicates should be snake_case verbs (e.g., obscures, encodes_via, overwhelms, requires_expertise_for)
- Base everything on what the participant comments actually describe

Output ONLY valid JSON matching this exact schema (no markdown, no explanation):
{
  "objects": [
    {"id": 1, "name": "object_name", "region": "region"}
  ],
  "attributes": [
    {"object_id": 1, "attr": "attribute_text", "sentiment": "+", "subtopic": "SubTopic Name"}
  ],
  "relationships": [
    {"subj": 1, "pred": "predicate", "obj": 2, "sentiment": "+", "subtopic": "SubTopic Name"}
  ]
}"""


def build_user_message(row, subtopics_text):
    return f"""Image: {row['imageName']}
VisType: {row['VisType']}
Normalized VC score (ground truth): {row['NormalizedVC']:.2f}

Participant comments:
{row['rawUserComments']}

Curated complexity phrases:
{row['humanCuratedPhrases']}

Valid subtopics for labeling:
{subtopics_text}

Extract the scene graph elements for this image."""


def load_existing(out_file):
    if out_file.exists():
        return json.loads(out_file.read_text(encoding='utf-8'))
    return {}


def save(data, out_file):
    out_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def extract_one(client, row, subtopics_text):
    """Call Claude API for a single image row. Returns parsed extraction dict."""
    user_msg = build_user_message(row, subtopics_text)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}   # prompt caching
            }
        ],
        messages=[
            {"role": "user", "content": user_msg}
        ]
    )

    raw = response.content[0].text.strip()
    # Strip any accidental markdown code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Process only first N images')
    parser.add_argument('--overwrite', action='store_true', help='Re-process already extracted images')
    args = parser.parse_args()

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print('ERROR: ANTHROPIC_API_KEY not set. Check your .env file.')
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    df = pd.read_csv(DATA_CSV)
    subtopics_df = pd.read_csv(SUB_CSV)
    subtopics_text = build_subtopics_text(subtopics_df)

    OUT_FILE.parent.mkdir(exist_ok=True)
    existing = load_existing(OUT_FILE)

    images = df.to_dict('records')
    if args.limit:
        images = images[:args.limit]

    to_process = [r for r in images if args.overwrite or r['imageName'] not in existing]
    skip_count = len(images) - len(to_process)

    print(f'Total images: {len(images)}')
    print(f'Already done: {skip_count}  |  To process: {len(to_process)}')
    print(f'Output: {OUT_FILE.resolve()}\n')

    ok, failed = 0, []

    for i, row in enumerate(to_process, 1):
        img = row['imageName']
        print(f'[{i}/{len(to_process)}] {img} (VC={row["NormalizedVC"]:.2f}) ... ', end='', flush=True)
        try:
            extraction = extract_one(client, row, subtopics_text)
            existing[img] = extraction
            save(existing, OUT_FILE)
            n_obj = len(extraction.get('objects', []))
            n_attr = len(extraction.get('attributes', []))
            n_rel = len(extraction.get('relationships', []))
            print(f'OK  ({n_obj} obj, {n_attr} attr, {n_rel} rel)')
            ok += 1
        except json.JSONDecodeError as e:
            print(f'JSON PARSE ERROR: {e}')
            failed.append(img)
        except anthropic.RateLimitError:
            print('RATE LIMITED — waiting 60s ...')
            time.sleep(60)
            failed.append(img)
        except Exception as e:
            print(f'ERROR: {e}')
            failed.append(img)

        if i < len(to_process):
            time.sleep(SLEEP_BETWEEN)

    print(f'\n=== Done: {ok} extracted, {len(failed)} failed ===')
    if failed:
        print('Failed images:')
        for f in failed:
            print(f'  {f}')
    print(f'Results saved to {OUT_FILE.resolve()}')


if __name__ == '__main__':
    main()
