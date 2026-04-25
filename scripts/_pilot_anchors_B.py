"""Pilot: run the new Condition B prompt (originalPhrases + sentiment tags)
on the 3 anchor images only, writing to a separate file so the production
oar_B.json is untouched.

Usage:
  python scripts/_pilot_anchors_B.py
"""
from __future__ import annotations
import os, sys, json, asyncio
from pathlib import Path
import importlib.util

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / 'scripts' / '_extract_three_conditions.py'

spec = importlib.util.spec_from_file_location('three_conditions', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

OUT_FILE = m.OUT_DIR / 'oar_B_anchor_pilot.json'


async def main():
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print('ERROR: ANTHROPIC_API_KEY not set'); sys.exit(1)

    import anthropic, aiohttp
    client = anthropic.AsyncAnthropic(api_key=api_key)

    rows_all = m.load_eval_rows(include_anchors=True)
    anchor_names = m.ANCHOR_NAMES
    rows = [r for r in rows_all if r['imageName'] in anchor_names]
    if len(rows) != 3:
        print(f'ERROR: expected 3 anchors, got {len(rows)}'); sys.exit(1)

    print(f'Running Condition B (new prompt) on {len(rows)} anchors')
    print(f'Model: {m.MODEL}  T={m.TEMPERATURE}  max_tokens={m.MAX_TOKENS}')
    print()

    out = {}
    async with aiohttp.ClientSession() as session:
        for row in rows:
            name = row['imageName']
            print(f'--- {name} ---')
            print('Prompt phrases:')
            print(m.format_tagged_phrases(row))
            print()
            try:
                ext = await m.extract_one(client, session, row, 'B', m.SYSTEM_B, [])
                out[name] = ext
                print(f'OK  objects={len(ext.get("objects",[]))} '
                      f'attrs={len(ext.get("attributes",[]))} '
                      f'rels={len(ext.get("relationships",[]))}')
                print(json.dumps(ext, indent=2, ensure_ascii=False))
            except Exception as e:
                print(f'ERROR: {e}')
            print()

    OUT_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Wrote {OUT_FILE}')


if __name__ == '__main__':
    asyncio.run(main())
