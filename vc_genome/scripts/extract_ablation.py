"""
Prompt Ablation Extraction — v0, v1, v2, v3, v4
=================================================
Runs the 27-image subset through stripped-down prompt variants:
  v0: image only → topic selection
  v1: image only → scene graph (element-level)
  v2: image + VisType label → scene graph (element-level)
  v3: image + original phrases → scene graph (element-level)
  v4: image + original phrases → scene graph (pattern-level)

Usage:
    python scripts/extract_ablation.py v0          # image only (topics)
    python scripts/extract_ablation.py v1          # image only (scene graph)
    python scripts/extract_ablation.py v2          # image + vistype
    python scripts/extract_ablation.py v3          # image + phrases (elements)
    python scripts/extract_ablation.py v4          # image + phrases (patterns)
    python scripts/extract_ablation.py v3 v4      # run multiple
    python scripts/extract_ablation.py v4 --overwrite  # re-process
    python scripts/extract_ablation.py v4 --concurrency 3
"""

import os, sys, json, asyncio, argparse, base64
from pathlib import Path
import aiohttp
import pandas as pd
from dotenv import load_dotenv
import anthropic

# ── Config ──────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

SUBSET_CSV = Path('vc_genome_output_full/subset_27.csv')
OUT_DIR    = Path('vc_genome_output_full/ablation')
MODEL      = 'claude-sonnet-4-6'
MAX_TOKENS = 4096
CONCURRENCY = 5

IMAGE_BASE_URL = "https://raw.githubusercontent.com/c109363/ExperimentImage/main/AllDataResize/"

# ── System prompts per version ──────────────────────────────────────────────

SYSTEM_V0 = """You are a data visualization expert. You will receive an image of a data visualization.

Your task: Identify which visual complexity topics are most relevant to this visualization, based purely on what you observe in the image.

## Visual Complexity Topics

1. **Data Density / Image Clutter** — The perceived amount, richness, or depth of data content. Considers information volume, element quantity, and visual clutter/overlap.
2. **Visual Encoding Clarity** — The variety, type, and complexity of graphical forms (shapes, lines, marks) and how spatial layout, scale, and encoding interpretability contribute to complexity.
3. **Semantics / Text Legibility** — The quantity and density of text elements (titles, axis labels, legends, captions, annotations, in-chart labels).
4. **Schema** — Whether specialized domain knowledge is needed, including dimensionality (2D/3D), structural complexity, and abstraction level.
5. **Color, Symbol, and Texture Details** — Range, variety, and arrangement of colors, plus use of symbols, textures, and non-color graphical markers.
6. **Aesthetics Uncertainty** — How visually cluttered, dense, or disordered the layout appears.
7. **Immediacy / Cognitive Load** — Overall ease or difficulty of interpreting the visualization. Considers interpretive difficulty, semantic clarity, and processing time/effort.

## Instructions
- Select up to three topics (by name) that are most relevant to the visual complexity of this image. List them in order of relevance, most relevant first.
- Write 1–3 sentences explaining why these topics are relevant to this visualization, in the order of relevance.

Output ONLY valid JSON matching this schema (no markdown, no explanation):
{
  "topics": [
    {"topic": "topic_name"}
  ],
  "explanation": "1-3 concise sentences (max 100 words total) explaining why these topics are relevant."
}"""

SYSTEM_V1 = """You are a visual complexity annotation expert. You will receive an image of a data visualization.

Your task: Extract a scene graph (objects, attributes, relationships) describing the visual complexity of this visualization based purely on what you observe in the image.

Rules:
- Extract only what is visible in the image. Do not invent or assume elements that are not present.
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
    {"object_id": 1, "attr": "attribute_text", "sentiment": "+/-"}
  ],
  "relationships": [
    {"subj": 1, "pred": "predicate", "obj": 2, "sentiment": "+/-"}
  ]
}"""

SYSTEM_V2 = """You are a visual complexity annotation expert. You will receive:
1. An image of a data visualization
2. The visualization type (e.g., Bar, Line, Node-link)

Your task: Extract a scene graph (objects, attributes, relationships) describing the visual complexity of this visualization based on what you observe, given its type.

Rules:
- Extract only what is visible in the image. Do not invent or assume elements that are not present.
- Object names: single lowercase words or snake_case (e.g., axis_label, legend, gridline)
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
    {"object_id": 1, "attr": "attribute_text", "sentiment": "+/-"}
  ],
  "relationships": [
    {"subj": 1, "pred": "predicate", "obj": 2, "sentiment": "+/-"}
  ]
}"""

SYSTEM_V3 = """You are a visual complexity annotation expert. You will receive:
1. An image of a data visualization
2. Original phrases extracted from participant comments describing their perception of its visual complexity

Your task: Extract a scene graph (objects, attributes, relationships) that is grounded in what participants mentioned AND what is visible in the image.

Rules:
- Use the phrases to guide what to extract. Each phrase highlights a visual element or quality that participants noticed.
- Use the image to correctly identify what visual element each phrase refers to.
- Do not invent objects, attributes, or relationships that are not supported by the phrases or visible in the image.
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
    {"object_id": 1, "attr": "attribute_text", "sentiment": "+/-"}
  ],
  "relationships": [
    {"subj": 1, "pred": "predicate", "obj": 2, "sentiment": "+/-"}
  ]
}"""

SYSTEM_V4 = """You are a visual complexity annotation expert. You will receive:
1. An image of a data visualization
2. Original phrases extracted from participant comments describing their perception of its visual complexity

Your task: Extract a **pattern-level** scene graph describing the overall visual patterns and compositional structures that drive complexity, grounded in what participants mentioned AND what is visible in the image.

Focus on general patterns rather than individual elements:
- Objects should represent visual patterns, groupings, or compositional structures (e.g., "dense_cluster", "overlapping_layers", "color_gradient_pattern", "repeated_grid") rather than individual marks or labels.
- Attributes should describe pattern-level qualities (e.g., "high_density", "irregular_spacing", "competing_visual_channels") rather than properties of single elements.
- Relationships should capture how patterns interact to create complexity (e.g., dense_cluster "competes_with" label_pattern) rather than element-to-element connections.

Rules:
- Use the phrases to identify which visual patterns participants perceived as contributing to complexity.
- Use the image to verify and ground those patterns.
- Do not list individual marks, labels, or data points as separate objects. Aggregate them into patterns.
- Object names: snake_case pattern descriptors (e.g., dense_mark_cluster, overlapping_text_block, multi_color_encoding)
- Regions: data_area, axes, legend, title, annotation, overall
- Attribute text: snake_case, max 4 words
- Predicates: snake_case verbs describing pattern interactions (e.g., competes_with, amplifies, fragments, overwhelms)
- Sentiment: "+" if pattern increases perceived complexity, "-" if decreases

Output ONLY valid JSON matching this schema (no markdown, no explanation):
{
  "objects": [
    {"id": 1, "name": "pattern_name", "region": "region"}
  ],
  "attributes": [
    {"object_id": 1, "attr": "pattern_quality", "sentiment": "+/-"}
  ],
  "relationships": [
    {"subj": 1, "pred": "predicate", "obj": 2, "sentiment": "+/-"}
  ]
}"""

SYSTEM_PROMPTS = {'v0': SYSTEM_V0, 'v1': SYSTEM_V1, 'v2': SYSTEM_V2, 'v3': SYSTEM_V3, 'v4': SYSTEM_V4}

# ── User message builders per version ───────────────────────────────────────

def build_user_message_v0(row):
    return "Identify the most relevant visual complexity topics for this visualization."

def build_user_message_v1(row):
    return "Extract the scene graph for this data visualization based on what you observe."

def build_user_message_v2(row):
    return f"""Visualization type: {row['VisType']}

Extract the scene graph for this data visualization."""

def build_user_message_v3(row):
    return f"""Original phrases from participant comments:
{row['originalPhrases']}

Extract the scene graph elements grounded in these phrases and what you see in the image."""

def build_user_message_v4(row):
    return f"""Original phrases from participant comments:
{row['originalPhrases']}

Extract pattern-level scene graph elements — focus on visual patterns and compositional structures rather than individual elements."""

USER_BUILDERS = {'v0': build_user_message_v0, 'v1': build_user_message_v1, 'v2': build_user_message_v2, 'v3': build_user_message_v3, 'v4': build_user_message_v4}

# ── Shared infrastructure ──────────────────────────────────────────────────

def build_system_block(version):
    return [{"type": "text", "text": SYSTEM_PROMPTS[version], "cache_control": {"type": "ephemeral"}}]

async def fetch_image_base64(session, image_name):
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

async def extract_one(client, session, row, system_block, user_text):
    img_name = row['imageName']
    image_b64 = await fetch_image_base64(session, img_name)

    response = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_block,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                {"type": "text", "text": user_text}
            ]
        }]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return img_name, json.loads(raw)

async def process_one(client, session, row, system_block, user_builder, semaphore, results, lock, out_file, counter):
    async with semaphore:
        img_name = row['imageName']
        user_text = user_builder(row)
        try:
            name, extraction = await extract_one(client, session, row, system_block, user_text)

            # Summarize output depending on schema
            if 'topics' in extraction:
                topics = [t.get('topic', '?') for t in extraction['topics']]
                summary = f"topics: {', '.join(topics)}"
            else:
                n_obj = len(extraction.get('objects', []))
                n_attr = len(extraction.get('attributes', []))
                n_rel = len(extraction.get('relationships', []))
                summary = f"{n_obj} obj, {n_attr} attr, {n_rel} rel"

            async with lock:
                results[name] = extraction
                counter['ok'] += 1
                idx = counter['ok'] + counter['fail']
                save(results, out_file)

            print(f'  [{idx}/{counter["total"]}] {img_name} OK  ({summary})')

        except json.JSONDecodeError as e:
            async with lock:
                counter['fail'] += 1
                idx = counter['ok'] + counter['fail']
            print(f'  [{idx}/{counter["total"]}] {img_name} JSON PARSE ERROR: {e}')

        except anthropic.RateLimitError:
            async with lock:
                counter['fail'] += 1
                idx = counter['ok'] + counter['fail']
            print(f'  [{idx}/{counter["total"]}] {img_name} RATE LIMITED')

        except Exception as e:
            async with lock:
                counter['fail'] += 1
                idx = counter['ok'] + counter['fail']
            print(f'  [{idx}/{counter["total"]}] {img_name} ERROR: {e}')

async def run_version(client, version, df, args):
    out_file = OUT_DIR / f'llm_extractions_{version}.json'
    existing = load_existing(out_file)

    system_block = build_system_block(version)
    user_builder = USER_BUILDERS[version]

    images = df.to_dict('records')
    if args.images:
        target_set = set(args.images)
        images = [r for r in images if r['imageName'] in target_set]
    to_process = [r for r in images if args.overwrite or r['imageName'] not in existing]
    skip_count = len(images) - len(to_process)

    print(f'\n{"="*50}')
    print(f'Version: {version.upper()}')
    print(f'  Total: {len(images)}  |  Skip: {skip_count}  |  Process: {len(to_process)}')
    print(f'  Output: {out_file}')

    if not to_process:
        print('  Nothing to process.')
        return

    concurrency = args.concurrency or CONCURRENCY
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    counter = {'ok': 0, 'fail': 0, 'total': len(to_process)}

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_one(client, session, row, system_block, user_builder, semaphore, existing, lock, out_file, counter)
            for row in to_process
        ]
        await asyncio.gather(*tasks)

    print(f'  Done. Success: {counter["ok"]}  |  Failed: {counter["fail"]}')

async def async_main(args):
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print('ERROR: ANTHROPIC_API_KEY not set. Check your .env file.')
        sys.exit(1)

    client = anthropic.AsyncAnthropic(api_key=api_key)
    df = pd.read_csv(SUBSET_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Loaded {len(df)} images from {SUBSET_CSV}')
    print(f'Model: {MODEL}')

    for version in args.versions:
        await run_version(client, version, df, args)

def main():
    parser = argparse.ArgumentParser(description='Run prompt ablation extractions (v0/v1/v2)')
    parser.add_argument('versions', nargs='+', choices=['v0', 'v1', 'v2', 'v3', 'v4'], help='Which prompt version(s) to run')
    parser.add_argument('--overwrite', action='store_true', help='Re-process already extracted images')
    parser.add_argument('--images', nargs='+', help='Process only these specific image names')
    parser.add_argument('--concurrency', type=int, default=None, help=f'Max parallel API calls (default: {CONCURRENCY})')
    args = parser.parse_args()

    # Set working directory to project root
    os.chdir(Path(__file__).parent.parent)
    asyncio.run(async_main(args))

if __name__ == '__main__':
    main()
