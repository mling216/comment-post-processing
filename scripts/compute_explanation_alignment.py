"""
LLM Explanation vs Human Comment alignment.

For each image we have:
  - LLM explanation text (Prominent prompt; image-level rationale).
  - Zero or more human free-text comments collected during pairwise study.

Three primary metrics per image:
  (1) Chart-part role Jaccard:
      Build a mention set of canonical chart-part roles from each text using a
      keyword lexicon; compute Jaccard between LLM and human-union sets.
  (2) TF-IDF cosine similarity:
      Sklearn TfidfVectorizer fit on the joint corpus; cosine between LLM text
      and the concatenation of human comments for that image.
  (3) Style profile:
      tokens, hedging-rate, first-person-rate, causal-connective rate.

Plus a human-human ceiling for (1) and (2) on images with >=2 human comments
(leave-one-out: each rater vs union of the other raters).

Random-pair baseline for (1) and (2): shuffle the human pool against LLM
explanations and recompute, giving a chance level reference.

Output: results/alignment_explanation_vs_comment.csv (per-image)
        plus stdout summary tables.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]

# Files
LLM_63   = ROOT / "results" / "probe_prominent_63"  / "prominent_results.csv"
LLM_510  = ROOT / "results" / "probe_prominent_510_opus_4_6" / "prominent_results.csv"
HUMAN_PQ = ROOT / "Claude_vc_prediction" / "ResultsStepByStep - 0.postquestionare_all.csv"

ANCHORS = {
    "VisC.503.6.png",
    "InfoVisJ.619.17.png",
    "InfoVisJ.1149.6(1).png",
}

SEED = 42
N_BOOT = 2000

# =====================================================================
# Lexicons
# =====================================================================

# 8 chart-part roles + a synonym lexicon (matches roles in app:oar).
ROLE_LEXICON: dict[str, list[str]] = {
    "chart":       ["chart", "graph", "plot", "figure", "visualization", "visualisation", "diagram"],
    "data_area":   ["bar", "bars", "line", "lines", "curve", "curves", "point", "points",
                    "marker", "markers", "scatter", "wedge", "slice", "node", "edge", "cell",
                    "tile", "ribbon", "area", "stack", "polygon", "glyph", "icon", "shape",
                    "data point", "datapoint"],
    "encoding":    ["color", "colour", "hue", "shade", "shading", "gradient", "palette", "fill",
                    "size", "width", "height", "thickness", "opacity", "transparency",
                    "position", "encoding", "symbol", "symbols", "pattern", "texture"],
    "title":       ["title", "subtitle", "header", "heading", "caption"],
    "axes":        ["axis", "axes", "x-axis", "y-axis", "tick", "ticks", "scale", "scales",
                    "gridline", "grid line", "grid", "grids", "log scale"],
    "labels":      ["label", "labels", "annotation", "annotations", "callout", "callouts",
                    "text", "wording", "word", "letter", "letters", "number", "numbers",
                    "value", "values", "data label", "tooltip"],
    "legend":      ["legend", "key", "color key", "legend entry", "legend item"],
    "background":  ["background", "backdrop", "frame", "border", "panel", "whitespace",
                    "negative space"],
}

ROLES = list(ROLE_LEXICON.keys())

# Compile lookup phrase -> role
_PHRASE_TO_ROLE: list[tuple[re.Pattern, str]] = []
for role, words in ROLE_LEXICON.items():
    for w in sorted(set(words), key=len, reverse=True):
        # word-boundary match; allow plural/singular by base form
        pat = re.compile(r"\b" + re.escape(w) + r"s?\b", re.IGNORECASE)
        _PHRASE_TO_ROLE.append((pat, role))

# Style markers
HEDGE_WORDS = {"may", "might", "could", "would", "perhaps", "possibly", "likely",
               "appears", "appear", "seems", "seem", "suggests", "suggest",
               "tends", "tend", "somewhat", "relatively", "moderately", "slightly",
               "fairly", "rather"}
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "our", "us"}
CAUSAL = {"because", "since", "due", "thus", "therefore", "so", "hence",
          "consequently", "as a result", "owing to", "leads to", "leading to",
          "causes", "cause", "drives", "driving", "creates"}

# =====================================================================
# Text cleaning / extraction
# =====================================================================

_TOK = re.compile(r"[A-Za-z][A-Za-z\-']+")

def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOK.findall(text or "")]

def role_mentions(text: str) -> set[str]:
    """Return the set of role keys mentioned anywhere in `text`."""
    if not text:
        return set()
    s = text.lower()
    found = set()
    for pat, role in _PHRASE_TO_ROLE:
        if pat.search(s):
            found.add(role)
    return found

def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return float("nan")
    if not (a | b):
        return float("nan")
    return len(a & b) / len(a | b)

def style_profile(text: str) -> dict[str, float]:
    toks = tokens(text)
    n = max(len(toks), 1)
    text_lc = (text or "").lower()
    return dict(
        n_tokens   = len(toks),
        hedge_rate = sum(t in HEDGE_WORDS for t in toks) / n * 100,
        first_pers = sum(t in FIRST_PERSON for t in toks) / n * 100,
        causal_rate= sum(any(c in text_lc for c in [" " + cw + " "])
                          for cw in CAUSAL) / n * 100 * 0  # replaced below
    )

def causal_rate(text: str) -> float:
    """Causal-connective rate per 100 tokens (simple substring count)."""
    toks = tokens(text)
    if not toks:
        return 0.0
    text_lc = " " + (text or "").lower() + " "
    hits = 0
    for cw in CAUSAL:
        # whole-word for single tokens, substring for phrases
        if " " in cw:
            hits += text_lc.count(" " + cw + " ")
        else:
            hits += sum(1 for t in toks if t == cw)
    return hits / len(toks) * 100

def style_dict(text: str) -> dict[str, float]:
    toks = tokens(text)
    n = max(len(toks), 1)
    return {
        "n_tokens":   len(toks),
        "hedge_rate": sum(t in HEDGE_WORDS for t in toks) / n * 100,
        "first_pers": sum(t in FIRST_PERSON for t in toks) / n * 100,
        "causal_rate": causal_rate(text),
    }

# =====================================================================
# Load data
# =====================================================================

def load_human_comments() -> dict[str, list[str]]:
    """imageName -> list of human comment strings (one per response)."""
    df = pd.read_csv(HUMAN_PQ)
    pool: dict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        for img_col, c_col in [
            ("MoreComplexImageName", "CommentCollectedOnMoreComplexFig"),
            ("LessComplexImageName", "CommentCollectedOnLessComplexFig"),
        ]:
            img = row.get(img_col)
            cmt = row.get(c_col)
            if pd.isna(img) or pd.isna(cmt):
                continue
            cmt = str(cmt).strip()
            if cmt and cmt.lower() not in {"nan", "none", "n/a", "na"}:
                pool[str(img)].append(cmt)
    return pool

def load_llm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"filename": "imageName"})
    df = df[~df["imageName"].isin(ANCHORS)].copy()
    df["explanation"] = df["explanation"].fillna("").astype(str)
    return df.reset_index(drop=True)

# =====================================================================
# Core computation
# =====================================================================

def compute_per_image(llm_df: pd.DataFrame, human: dict[str, list[str]]) -> pd.DataFrame:
    """Compute alignment metrics per LLM image; only includes images with >=1 human comment."""
    rows = []
    for _, row in llm_df.iterrows():
        img = row["imageName"]
        h_list = human.get(img, [])
        if not h_list:
            continue
        llm_text = row["explanation"]
        h_text = " \n".join(h_list)

        rec = {
            "imageName":   img,
            "n_human":     len(h_list),
            "llm_roles":   ";".join(sorted(role_mentions(llm_text))),
            "human_roles": ";".join(sorted(role_mentions(h_text))),
            "role_jaccard": jaccard(role_mentions(llm_text), role_mentions(h_text)),
        }
        # style for LLM and human (concat)
        for k, v in style_dict(llm_text).items():
            rec["llm_" + k] = v
        for k, v in style_dict(h_text).items():
            rec["human_" + k] = v
        rec["llm_text"]   = llm_text
        rec["human_text"] = h_text
        rows.append(rec)
    return pd.DataFrame(rows)


def add_tfidf_cosine(per_img: pd.DataFrame, llm_df: pd.DataFrame,
                     human: dict[str, list[str]]) -> pd.DataFrame:
    """Fit TF-IDF on the joint corpus and add cosine for each row."""
    docs = list(per_img["llm_text"]) + list(per_img["human_text"])
    vec = TfidfVectorizer(lowercase=True, stop_words="english",
                          ngram_range=(1, 2), min_df=2, max_df=0.95)
    M = vec.fit_transform(docs)
    n = len(per_img)
    L, H = M[:n], M[n:]
    full = cosine_similarity(L, H)
    sims = np.diag(full)
    per_img = per_img.copy()
    per_img["tfidf_cosine"] = sims
    # also store fitted vectorizer artefacts for chance baseline
    per_img.attrs["L"] = L
    per_img.attrs["H"] = H
    return per_img


def chance_baseline(per_img: pd.DataFrame, n_perms: int = 200) -> dict[str, float]:
    """Random-pair (LLM_i vs Human_pi(i)) cosine and role-jaccard chance.
    Vectorised: compute full LxH cosine matrix once, then permutations are
    diagonal selections of the permuted matrix."""
    rng = np.random.default_rng(SEED)
    L = per_img.attrs["L"]
    H = per_img.attrs["H"]
    n = L.shape[0]

    # full nxn cosine matrix between LLM and Human docs
    full = cosine_similarity(L, H)  # (n, n)

    llm_roles   = [set(s.split(";")) - {""} for s in per_img["llm_roles"]]
    human_roles = [set(s.split(";")) - {""} for s in per_img["human_roles"]]

    cos_means: list[float] = []
    jac_means: list[float] = []
    for _ in range(n_perms):
        perm = rng.permutation(n)
        cos_means.append(float(full[np.arange(n), perm].mean()))
        jvals = [jaccard(llm_roles[i], human_roles[perm[i]]) for i in range(n)]
        jvals = [v for v in jvals if not np.isnan(v)]
        jac_means.append(float(np.mean(jvals)) if jvals else float("nan"))
    return {
        "cos_chance_mean": float(np.mean(cos_means)),
        "cos_chance_lo":   float(np.percentile(cos_means, 2.5)),
        "cos_chance_hi":   float(np.percentile(cos_means, 97.5)),
        "jac_chance_mean": float(np.nanmean(jac_means)),
        "jac_chance_lo":   float(np.nanpercentile(jac_means, 2.5)),
        "jac_chance_hi":   float(np.nanpercentile(jac_means, 97.5)),
    }


def human_human_ceiling(human: dict[str, list[str]],
                         llm_images: set[str]) -> dict[str, float]:
    """Leave-one-out per-rater vs union of others on images with >=2 raters
    (restricted to LLM-evaluated images)."""
    pairs_role: list[float] = []
    pairs_cos:  list[float] = []
    eligible = [(img, lst) for img, lst in human.items()
                if img in llm_images and len(lst) >= 2]
    if not eligible:
        return {"n_images": 0}

    # Build TF-IDF over the union of all comments for these images.
    all_docs: list[str] = []
    doc_keys: list[tuple[str, int]] = []  # (image, rater idx)
    for img, lst in eligible:
        for i, c in enumerate(lst):
            all_docs.append(c)
            doc_keys.append((img, i))
    vec = TfidfVectorizer(lowercase=True, stop_words="english",
                          ngram_range=(1, 2), min_df=2, max_df=0.95)
    if len(all_docs) < 2:
        return {"n_images": 0}
    M = vec.fit_transform(all_docs)
    # index by (img,i)
    idx = {key: k for k, key in enumerate(doc_keys)}

    for img, lst in eligible:
        roles = [role_mentions(c) for c in lst]
        for i, c in enumerate(lst):
            others = lst[:i] + lst[i + 1:]
            other_roles = set().union(*[role_mentions(o) for o in others])
            j = jaccard(roles[i], other_roles)
            if not np.isnan(j):
                pairs_role.append(j)
            # cosine: rater i vs union (concatenated then re-vectorised for fair comparison)
            other_text = " \n".join(others)
            v_other = vec.transform([other_text])
            cos = float(cosine_similarity(M[idx[(img, i)]], v_other)[0, 0])
            pairs_cos.append(cos)

    return {
        "n_images": len(eligible),
        "n_raters": len(pairs_cos),
        "ceiling_jaccard": float(np.mean(pairs_role)),
        "ceiling_cosine":  float(np.mean(pairs_cos)),
    }


def boot_mean_ci(x: np.ndarray, n_boot: int = N_BOOT, alpha: float = 0.05) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.array([rng.choice(x, n, replace=True).mean() for _ in range(n_boot)])
    return float(np.mean(x)), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


# =====================================================================
# Reporting
# =====================================================================

def report(label: str, per_img: pd.DataFrame, ceiling: dict, chance: dict) -> dict:
    print(f"\n=================================================================")
    print(f"  {label}")
    print(f"=================================================================")
    n = len(per_img)
    print(f"  n_images with paired LLM+human text: {n}")

    # role jaccard
    jac = per_img["role_jaccard"].dropna().values
    jm, jlo, jhi = boot_mean_ci(jac)
    print(f"\n  Chart-part role Jaccard (LLM vs human-union):")
    print(f"    mean = {jm:.3f}  95%CI = [{jlo:.3f}, {jhi:.3f}]   (n={len(jac)})")
    print(f"    chance baseline mean = {chance['jac_chance_mean']:.3f}  "
          f"95% range [{chance['jac_chance_lo']:.3f}, {chance['jac_chance_hi']:.3f}]")
    if ceiling.get("n_images"):
        print(f"    human-human ceiling = {ceiling['ceiling_jaccard']:.3f}  "
              f"(leave-one-out, {ceiling['n_images']} images, {ceiling['n_raters']} raters)")
        ratio_j = jm / ceiling["ceiling_jaccard"] if ceiling["ceiling_jaccard"] > 0 else float("nan")
        print(f"    LLM / ceiling = {ratio_j:.2f}")

    # cosine
    cos = per_img["tfidf_cosine"].dropna().values
    cm, clo, chi = boot_mean_ci(cos)
    print(f"\n  TF-IDF cosine similarity (LLM vs human-union):")
    print(f"    mean = {cm:.3f}  95%CI = [{clo:.3f}, {chi:.3f}]   (n={len(cos)})")
    print(f"    chance baseline mean = {chance['cos_chance_mean']:.3f}  "
          f"95% range [{chance['cos_chance_lo']:.3f}, {chance['cos_chance_hi']:.3f}]")
    if ceiling.get("n_images"):
        print(f"    human-human ceiling = {ceiling['ceiling_cosine']:.3f}")
        ratio_c = cm / ceiling["ceiling_cosine"] if ceiling["ceiling_cosine"] > 0 else float("nan")
        print(f"    LLM / ceiling = {ratio_c:.2f}")

    # style deltas
    print(f"\n  Style profile (mean per image):")
    print(f"    {'metric':<14} {'LLM':>8} {'Human':>8} {'delta':>8}")
    for key, name in [("n_tokens", "n_tokens"), ("hedge_rate", "hedge%"),
                      ("first_pers", "first-pers%"), ("causal_rate", "causal%")]:
        l = per_img["llm_"   + key].mean()
        h = per_img["human_" + key].mean()
        print(f"    {name:<14} {l:>8.2f} {h:>8.2f} {l-h:>+8.2f}")

    return {
        "label": label, "n": n,
        "jac_mean": jm, "jac_lo": jlo, "jac_hi": jhi,
        "jac_chance": chance["jac_chance_mean"],
        "jac_ceiling": ceiling.get("ceiling_jaccard", float("nan")),
        "cos_mean": cm, "cos_lo": clo, "cos_hi": chi,
        "cos_chance": chance["cos_chance_mean"],
        "cos_ceiling": ceiling.get("ceiling_cosine", float("nan")),
        "ceiling_n_images": ceiling.get("n_images", 0),
    }


def main():
    print("Loading data...")
    human = load_human_comments()
    print(f"  human comments: {sum(len(v) for v in human.values())} responses "
          f"across {len(human)} images")

    summaries = []
    out_rows = []
    for label, path in [("63-image (Prominent Opus)",  LLM_63),
                        ("510-image (Prominent Opus)", LLM_510)]:
        if not path.exists():
            print(f"  SKIP {label}: {path} missing")
            continue
        llm_df = load_llm(path)
        per_img = compute_per_image(llm_df, human)
        per_img = add_tfidf_cosine(per_img, llm_df, human)
        chance = chance_baseline(per_img, n_perms=200)
        ceiling = human_human_ceiling(human, set(per_img["imageName"]))
        s = report(label, per_img, ceiling, chance)
        summaries.append(s)

        per_img2 = per_img.drop(columns=["llm_text", "human_text"], errors="ignore")
        per_img2.insert(0, "set", label)
        out_rows.append(per_img2)

    if out_rows:
        out_df = pd.concat(out_rows, ignore_index=True)
        out_dir = ROOT / "results"
        out_dir.mkdir(exist_ok=True)
        out_csv = out_dir / "alignment_explanation_vs_comment.csv"
        out_df.to_csv(out_csv, index=False)
        print(f"\nSaved per-image -> {out_csv}")

        sum_csv = out_dir / "alignment_explanation_summary.csv"
        pd.DataFrame(summaries).to_csv(sum_csv, index=False)
        print(f"Saved summary   -> {sum_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
