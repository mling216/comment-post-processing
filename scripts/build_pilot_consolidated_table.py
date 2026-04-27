"""Build a consolidated per-image table for the 46 pilot-study images.

Columns:
    imageName, imageURL, isAnchor, VisType, NormalizedVC,
    originalPhrases (compiled human phrases),
    human_topics (canonical 7-topic keys),
    llm_topics_V0T, llm_topics_V0TW, llm_topics_V1_t0,
    OAR_B, OAR_V1, OAR_V2

Each OAR_* column is a compact text summary:
    objs: name1; name2; ...
    attrs: obj_name:attr[+/-]; ...
    rels: subj_name -pred-> obj_name [+/-]; ...

Run:
    python scripts/build_pilot_consolidated_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "tables" / "pilot_study_consolidated.csv"

ANCHORS = {"VisC.503.6.png", "InfoVisJ.619.17.png", "InfoVisJ.1149.6(1).png"}

TOPIC_KEYS = [
    "data_density",
    "visual_encoding",
    "text_annotation",
    "domain_schema",
    "color_symbol",
    "aesthetic_order",
    "cognitive_load",
]


def load_compiled() -> pd.DataFrame:
    p = ROOT / "phrase_reduction_v2" / "image_compiled_phrases.csv"
    df = pd.read_csv(p)
    gt = pd.read_csv(ROOT / "Claude_vc_prediction" / "gt_all_66.csv")
    pilot = set(gt["imageName"])
    df = df[df["imageName"].isin(pilot)].copy()
    return df[["imageName", "imageURL", "VisType", "NormalizedVC", "originalPhrases"]]


def load_human_topics() -> pd.DataFrame:
    p = ROOT / "topic_selection" / "human_topic_gt.csv"
    df = pd.read_csv(p)
    return df.rename(columns={"filename": "imageName", "human_topics": "human_topics"})[
        ["imageName", "human_topics"]
    ]


def load_topic_pred(folder: str, col: str) -> pd.DataFrame:
    p = ROOT / "results" / folder / "vc_scores.csv"
    df = pd.read_csv(p)
    out = df[["filename", "top3_topics"]].rename(
        columns={"filename": "imageName", "top3_topics": col}
    )
    return out


def derive_v1_top3() -> pd.DataFrame:
    p = ROOT / "results" / "vc_api_46gt_v1_t0" / "vc_scores.csv"
    df = pd.read_csv(p)
    rows = []
    for _, r in df.iterrows():
        scores = [(k, r[k]) for k in TOPIC_KEYS]
        scores.sort(key=lambda x: x[1], reverse=True)
        top3 = ";".join(k for k, _ in scores[:3])
        rows.append({"imageName": r["filename"], "llm_topics_V1_t0": top3})
    return pd.DataFrame(rows)


def fmt_oar(entry: dict) -> str:
    """Compact one-cell summary of an OAR extraction for one image."""
    if not entry:
        return ""
    objs = entry.get("objects", []) or []
    attrs = entry.get("attributes", []) or []
    rels = entry.get("relationships", []) or []

    id2name = {o.get("id"): o.get("name", "?") for o in objs}

    obj_part = "; ".join(o.get("name", "?") for o in objs)
    attr_part = "; ".join(
        f"{id2name.get(a.get('object_id'), '?')}:{a.get('attr', '?')}"
        f"[{a.get('sentiment', '?')}]"
        for a in attrs
    )
    rel_part = "; ".join(
        f"{id2name.get(r.get('subj'), '?')} -{r.get('pred', '?')}-> "
        f"{id2name.get(r.get('obj'), '?')} [{r.get('sentiment', '?')}]"
        for r in rels
    )
    return f"objs: {obj_part} | attrs: {attr_part} | rels: {rel_part}"


def load_oar(name: str, col: str) -> pd.DataFrame:
    p = ROOT / "vc_genome_output_full" / "three_conditions" / f"oar_{name}.json"
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    rows = [{"imageName": k, col: fmt_oar(v)} for k, v in data.items()]
    return pd.DataFrame(rows)


def main() -> None:
    base = load_compiled()
    base["isAnchor"] = base["imageName"].isin(ANCHORS)

    parts = [
        load_human_topics(),
        load_topic_pred("vc_api_topicsel_v0_t", "llm_topics_V0T"),
        load_topic_pred("vc_api_topicsel_v0_tw", "llm_topics_V0TW"),
        derive_v1_top3(),
        load_oar("B", "OAR_B"),
        load_oar("V1", "OAR_V1"),
        load_oar("V2", "OAR_V2"),
    ]

    out = base
    for p in parts:
        out = out.merge(p, on="imageName", how="left")

    out = out[
        [
            "imageName",
            "imageURL",
            "isAnchor",
            "VisType",
            "NormalizedVC",
            "originalPhrases",
            "human_topics",
            "llm_topics_V0T",
            "llm_topics_V0TW",
            "llm_topics_V1_t0",
            "OAR_B",
            "OAR_V1",
            "OAR_V2",
        ]
    ].sort_values(["isAnchor", "VisType", "imageName"]).reset_index(drop=True)

    # Normalise topic-list separator to "; " (was a bare ";" from upstream).
    for col in ("human_topics", "llm_topics_V0T", "llm_topics_V0TW", "llm_topics_V1_t0"):
        out[col] = (
            out[col]
            .astype("string")
            .map(lambda s: "; ".join(p.strip() for p in s.split(";")) if pd.notna(s) else s)
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")

    n_total = len(out)
    n_anchors = int(out["isAnchor"].sum())
    n_with_oar_B = int(out["OAR_B"].notna().sum())
    n_with_oar_V1 = int(out["OAR_V1"].notna().sum())
    n_with_oar_V2 = int(out["OAR_V2"].notna().sum())
    n_with_topics_V0T = int(out["llm_topics_V0T"].notna().sum())
    n_with_topics_V0TW = int(out["llm_topics_V0TW"].notna().sum())
    n_with_topics_V1 = int(out["llm_topics_V1_t0"].notna().sum())
    print(f"Wrote {OUT}")
    print(f"  rows: {n_total}  anchors: {n_anchors}")
    print(
        f"  OAR coverage   B={n_with_oar_B}  V1={n_with_oar_V1}  V2={n_with_oar_V2}"
    )
    print(
        f"  Topic coverage V0+T={n_with_topics_V0T}  V0+TW={n_with_topics_V0TW}  "
        f"V1(t=0)={n_with_topics_V1}"
    )


if __name__ == "__main__":
    main()
