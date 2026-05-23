"""
Claude Vision Stage 2 Extraction Script
========================================
Sends the actual image + participant comments to Claude Vision API,
producing grounded scene graph extractions without forced minimums.

Key differences from Stage 1:
  - Sends the image (via URL) so Claude can see what's actually there
  - No rigid extraction quotas — extracts only what comments support
  - More conservative output for sparse-comment images
  - Async concurrent API calls (configurable concurrency)
  - Prompt caching on system prompt + subtopics text

Usage:
    python _extract_stage2_vision.py                       # process all
    python _extract_stage2_vision.py --limit 10            # first N
    python _extract_stage2_vision.py --test                # 3 test images
    python _extract_stage2_vision.py --images img1.png img2.png  # specific
    python _extract_stage2_vision.py --overwrite           # re-process all
    python _extract_stage2_vision.py --concurrency 5       # parallel requests
"""

import os, sys, json, asyncio, argparse, base64
from pathlib import Path
import aiohttp
import pandas as pd
from dotenv import load_dotenv
import anthropic

# ── Config ──────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

DATA_CSV  = Path('phrase_reduction_v2/image_compiled_phrases.csv')
SUB_CSV   = Path('phrase_reduction_v2/phrase_shortlist.csv')
OUT_FILE  = Path('vc_genome_output_full/llm_extractions_vision.json')
MODEL     = 'claude-sonnet-4-6'
MAX_TOKENS = 1024
CONCURRENCY = 5        # max parallel API calls

# Base URL for fetching images
IMAGE_BASE_URL = "https://raw.githubusercontent.com/c109363/ExperimentImage/main/AllDataResize/"

# Test images: sparse comment, medium comment, rich comment
TEST_IMAGES = [
    "InfoVisJ.2412.7.png",   # "Simplistic symbol" — 2 words
    "VASTJ.1332.5.png",      # "two axes; fewer data points" — 5 words
    "visMost147.png",        # long multi-sentence comment
]

# ── Subtopics block (cached) ──────────────────────────────────────────────
def build_subtopics_text(subtopics_df):
    lines = []
    for _, r in subtopics_df.iterrows():
        lines.append(f"- {r['SubTopic']}: {r['Description']}")
    return '\n'.join(lines)

# ── System prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a visual complexity annotation expert. You will receive:
1. An image of a data visualization
2. Participant comments describing their perception of its visual complexity
3. Curated complexity phrases extracted from those comments

Your task: Extract a scene graph (objects, attributes, relationships) that is **strictly grounded** in what participants actually mentioned AND what is visible in the image.

Rules:
- Extract ONLY elements supported by the participant comments. If the comment says "Simplistic symbol", extract what "simplistic" and "symbol" refer to in the image — nothing more.
- Use the image to correctly identify what visual element the participant is referring to (e.g., if they say "symbol", look at the image to see what the symbol actually is).
- Do NOT invent objects, attributes, or relationships that the comments do not mention or imply.
- The number of extractions should be proportional to the richness of the comments:
  - Very short comments (1-3 words): 1-2 objects, 1-2 attributes, 0-1 relationships
  - Medium comments (1-2 sentences): 2-4 objects, 2-5 attributes, 1-3 relationships  
  - Rich comments (3+ sentences): 3-7 objects, 3-8 attributes, 2-5 relationships
- Object names: single lowercase words or snake_case (e.g., axis_label, symbol, color)
- Regions: data_area, axes, legend, title, annotation, overall
- Attribute text: snake_case, max 4 words
- Predicates: snake_case verbs (e.g., obscures, encodes_via, clutters)
- Sentiment: "+" if attribute/relationship increases perceived complexity, "-" if decreases

Output ONLY valid JSON matching this schema (no markdown, no explanation):
{
  "objects": [
    {"id": 1, "name": "object_name", "region": "region"}
  ],
  "attributes": [
    {"object_id": 1, "attr": "attribute_text", "sentiment": "+/-", "subtopic": "SubTopic Name"}
  ],
  "relationships": [
    {"subj": 1, "pred": "predicate", "obj": 2, "sentiment": "+/-", "subtopic": "SubTopic Name"}
  ]
}"""


def build_user_message(row, subtopics_text):
    return f"""Image: {row['imageName']}
VisType: {row['VisType']}
Normalized VC score: {row['NormalizedVC']:.2f}

Participant comments:
{row['rawUserComments']}

Curated complexity phrases:
{row['humanCuratedPhrases']}

Valid subtopics for labeling:
{subtopics_text}

Extract the scene graph elements grounded in the comments and what you see in the image."""


async def fetch_image_base64(session, image_name):
    """Fetch image from GitHub and return base64-encoded string."""
    url = IMAGE_BASE_URL + image_name
    async with session.get(url) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Failed to fetch image {url}: HTTP {resp.status}")
        data = await resp.read()
    return base64.standard_b64encode(data).decode('utf-8')


def load_existing(out_file):
    if out_file.exists():
        return json.loads(out_file.read_text(encoding='utf-8'))
    return {}


def save(data, out_file):
    out_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def build_system_block(subtopics_text):
    """Build system prompt with subtopics appended — cached as one block."""
    full_system = SYSTEM_PROMPT + f"\n\nAvailable subtopics taxonomy:\n{subtopics_text}"
    return [
        {
            "type": "text",
            "text": full_system,
            "cache_control": {"type": "ephemeral"}
        }
    ]


async def extract_one(client, session, row, system_block):
    """Call Claude Vision API for a single image. Returns (imageName, extraction_dict)."""
    img_name = row['imageName']
    user_text = build_user_message(row, "")  # subtopics already in system block
    image_b64 = await fetch_image_base64(session, img_name)

    response = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_block,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        }
                    },
                    {
                        "type": "text",
                        "text": user_text
                    }
                ]
            }
        ]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return img_name, json.loads(raw)


async def process_batch(client, session, rows, system_block, semaphore, results, lock, out_file, counter):
    """Process a single image with concurrency control."""
    async with semaphore:
        row = rows
        img_name = row['imageName']
        try:
            name, extraction = await extract_one(client, session, row, system_block)
            n_obj = len(extraction.get('objects', []))
            n_attr = len(extraction.get('attributes', []))
            n_rel = len(extraction.get('relationships', []))

            async with lock:
                results[name] = extraction
                counter['ok'] += 1
                idx = counter['ok'] + counter['fail']
                total = counter['total']
                save(results, out_file)

            print(f'[{idx}/{total}] {img_name} (VC={row["NormalizedVC"]:.2f}) OK  ({n_obj} obj, {n_attr} attr, {n_rel} rel)')

        except json.JSONDecodeError as e:
            async with lock:
                counter['fail'] += 1
                idx = counter['ok'] + counter['fail']
                total = counter['total']
            print(f'[{idx}/{total}] {img_name} JSON PARSE ERROR: {e}')
            counter['failed_names'].append(img_name)

        except anthropic.RateLimitError:
            async with lock:
                counter['fail'] += 1
                idx = counter['ok'] + counter['fail']
                total = counter['total']
            print(f'[{idx}/{total}] {img_name} RATE LIMITED — will retry next run')
            counter['failed_names'].append(img_name)

        except Exception as e:
            async with lock:
                counter['fail'] += 1
                idx = counter['ok'] + counter['fail']
                total = counter['total']
            print(f'[{idx}/{total}] {img_name} ERROR: {e}')
            counter['failed_names'].append(img_name)


async def async_main(args):
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print('ERROR: ANTHROPIC_API_KEY not set. Check your .env file.')
        sys.exit(1)

    client = anthropic.AsyncAnthropic(api_key=api_key)

    df = pd.read_csv(DATA_CSV)
    subtopics_df = pd.read_csv(SUB_CSV)
    subtopics_text = build_subtopics_text(subtopics_df)
    system_block = build_system_block(subtopics_text)

    OUT_FILE.parent.mkdir(exist_ok=True)
    existing = load_existing(OUT_FILE)

    # Determine which images to process
    if args.test:
        images = df[df['imageName'].isin(TEST_IMAGES)].to_dict('records')
        print(f'=== TEST MODE: {len(images)} images ===\n')
    elif args.images:
        images = df[df['imageName'].isin(args.images)].to_dict('records')
    else:
        images = df.to_dict('records')
        if args.limit:
            images = images[:args.limit]

    to_process = [r for r in images if args.overwrite or r['imageName'] not in existing]
    skip_count = len(images) - len(to_process)

    concurrency = args.concurrency or CONCURRENCY
    print(f'Total images: {len(images)}')
    print(f'Already done: {skip_count}  |  To process: {len(to_process)}')
    print(f'Concurrency: {concurrency}')
    print(f'Output: {OUT_FILE.resolve()}\n')

    if not to_process:
        print('Nothing to process.')
    else:
        semaphore = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()
        counter = {'ok': 0, 'fail': 0, 'total': len(to_process), 'failed_names': []}

        async with aiohttp.ClientSession() as session:
            tasks = [
                process_batch(client, session, row, system_block, semaphore, existing, lock, OUT_FILE, counter)
                for row in to_process
            ]
            await asyncio.gather(*tasks)

        print(f'\n{"="*50}')
        print(f'Done. Success: {counter["ok"]}  |  Failed: {counter["fail"]}')
        if counter['failed_names']:
            print(f'Failed images: {counter["failed_names"]}')

    # Print comparison for test/specific mode
    if args.test or args.images:
        target_names = TEST_IMAGES if args.test else args.images
        print(f'\n{"="*50}')
        print('RESULTS:')
        print('='*50)
        for img_name in target_names:
            if img_name in existing:
                ext = existing[img_name]
                row = df[df['imageName'] == img_name].iloc[0]
                print(f'\n--- {img_name} ---')
                print(f'  Comments: "{row["rawUserComments"]}"')
                print(f'  Objects ({len(ext["objects"])}): {[o["name"] for o in ext["objects"]]}')
                print(f'  Attributes ({len(ext["attributes"])}): {[a["attr"] for a in ext["attributes"]]}')
                print(f'  Relationships ({len(ext["relationships"])}): {[r["pred"] for r in ext["relationships"]]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Process only first N images')
    parser.add_argument('--overwrite', action='store_true', help='Re-process already extracted images')
    parser.add_argument('--test', action='store_true', help='Run on 3 test images only')
    parser.add_argument('--images', nargs='+', help='Specific image names to process')
    parser.add_argument('--concurrency', type=int, default=None, help=f'Max parallel API calls (default: {CONCURRENCY})')
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == '__main__':
    os.chdir(Path(__file__).parent.parent)
    main()
