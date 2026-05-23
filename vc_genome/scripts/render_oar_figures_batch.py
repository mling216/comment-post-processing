"""
Batch-render scene graphs for every image in the three-conditions OAR pool,
across all three conditions (B, V1, V2). PNG only.

Output (tracked):
    figures/scene_graphs/oar_<image>_{B,V1,V2}.png

Reuses the same `build_vg_graphviz` styling as `render_oar_figure.py`.
Skips images for which a given condition has no extraction (V1/V2 lack
the 3 anchor images by design).
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from _vc_canon import canonicalize  # noqa
from render_oar_figure import build_vg_graphviz  # noqa

IN_DIR = ROOT / "vc_genome_output_full" / "three_conditions"
OUT_DIR = ROOT / "figures" / "scene_graphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def render_one(image_name: str, condition: str, raw: dict) -> tuple[int, int, int]:
    canon = canonicalize(raw)
    dot = build_vg_graphviz(canon, image_name, condition)
    safe = image_name.removesuffix(".png")
    _stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        png_bytes = dot.pipe(format="png")
    finally:
        sys.stderr.close()
        sys.stderr = _stderr
    (OUT_DIR / f"oar_{safe}_{condition}.png").write_bytes(png_bytes)
    return len(canon["objects"]), len(canon["attributes"]), len(canon["relationships"])


def main() -> None:
    conds = {
        "B":  json.loads((IN_DIR / "oar_B.json").read_text(encoding="utf-8")),
        "V1": json.loads((IN_DIR / "oar_V1.json").read_text(encoding="utf-8")),
        "V2": json.loads((IN_DIR / "oar_V2.json").read_text(encoding="utf-8")),
    }
    all_images = sorted(set().union(*(d.keys() for d in conds.values())))
    print(f"Rendering {len(all_images)} images x 3 conditions -> {OUT_DIR}")
    counts = {"B": 0, "V1": 0, "V2": 0}
    skipped = 0
    for img in all_images:
        for cond, data in conds.items():
            if img not in data:
                continue
            safe = img.removesuffix(".png")
            out_path = OUT_DIR / f"oar_{safe}_{cond}.png"
            if out_path.exists():
                skipped += 1
                continue
            o, a, r = render_one(img, cond, data[img])
            counts[cond] += 1
        print(f"  {img}  done")
    print(f"Done. PNGs written: B={counts['B']}  V1={counts['V1']}  V2={counts['V2']}  skipped={skipped}")


if __name__ == "__main__":
    main()
