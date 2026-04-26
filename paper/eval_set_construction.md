# Evaluation Set Construction: 46 → 66 Images

*Documents the construction of the ground-truth evaluation set used in
`appendix_llm_evaluation.md`. The set grew from an initial 46 images
(23 pairs, §1 of the appendix) to 66 images (33 pairs) to achieve
balanced per-vis-type coverage. After excluding 3 anchor exemplars,
the usable test set is **63 images — 7 per vis-type across 9 types**.*

---

## 1. Original 46-Image Set (23 Pairs)

### 1.1 Construction

The 46 images were sub-sampled from approximately 700 paired images
collected in a prior pair-comparison study
(`ResultsStepByStep - 0.postquestionare_all.csv`). The pair study asked
participants to pick the more visually complex image in each pair; the
aggregated Bradley–Terry model yields a normalized VC rating $y \in [0,1]$
per image.

Pairs were ranked by the absolute VC difference $|\Delta\mathrm{VC}|$
between the two images, then binned across the full $|\Delta\mathrm{VC}|$
range. Roughly equal numbers of pairs were sampled from each bin, yielding
23 pairs (46 images) that span the full difficulty spectrum — from near-ties
($|\Delta\mathrm{VC}| \approx 0.02$, hard discrimination) to large-gap
pairs ($|\Delta\mathrm{VC}| \approx 0.50$, easy discrimination).

Ground-truth `NormalizedVC` ratings are sourced from
`phrase_reduction_v2/image_compiled_phrases.csv`. The set was constructed
to maximize $|\Delta\mathrm{VC}|$ coverage, **not** to be vis-type
balanced.

### 1.2 Anchor Exemplars

Three of the 46 images are reserved as few-shot anchor exemplars used in
all anchored prompt variants (V0+A, V0+TA, V0+TWA, V0+TWCA, and V1):

| Image | Vis Type | VC |
|---|---|---|
| `VisC.503.6.png` | Bar | 0.22 |
| `InfoVisJ.1149.6(1).png` | Grid | 0.54 |
| `InfoVisJ.619.17.png` | Point | 0.95 |

These three are excluded from all reported metrics, leaving **43 test
images** for the main ablation (§3–§9 of the appendix).

### 1.3 Vis-Type Distribution (46 images, 43 non-anchor)

| Vis Type | Total (incl. anchor) | Test (excl. anchor) |
|---|:---:|:---:|
| Bar | 8 | 7 |
| Area | 7 | 7 |
| Glyph | 7 | 7 |
| Grid | 6 | 5 |
| Node-link | 6 | 6 |
| Cont.-ColorPatn | 4 | 4 |
| Point | 3 | 2 |
| Text | 3 | 3 |
| Line | 2 | 2 |
| **Total** | **46** | **43** |

Bar, Area, and Glyph are over-represented; Point, Line, and
Cont.-ColorPatn are under-represented. Per-vis-type CCC estimates on the
43-image set are unreliable for Point (n=2), Line (n=2), and
Cont.-ColorPatn (n=4).

**Artifact:** `Claude_vc_prediction/gt_all_46.csv`

---

## 2. Expansion to 66 Images (10 New Pairs)

### 2.1 Motivation

The unbalanced distribution makes it impossible to draw reliable
per-type conclusions from the ablation. To address this, 10 additional
pairs were selected targeting the five under-represented types
(Grid, Node-link, Cont.-ColorPatn, Point, Text, Line) to bring every
type's non-anchor test count to exactly **7**, matching the already-full
Bar, Area, and Glyph counts.

### 2.2 Selection Criteria

Candidate pairs were drawn from the same pair-comparison pool
(`ResultsStepByStep - 0.postquestionare_all.csv`). A pair was eligible if:

1. **Both images have GT** — both filenames appear in
   `phrase_reduction_v2/image_compiled_phrases.csv` (ensures VC labels are
   available).
2. **Both images are new** — neither filename already appears in the
   46-image set.
3. **Vis-type contributes to an under-represented type** — at least one
   image in the pair belongs to a type that still needs more images.

Among eligible pairs, candidates were sorted by `diffScore` descending
(maximize pair difficulty spread). Selection proceeded greedily with a
**constrained-first** ordering: types furthest below the target count
were prioritized. All 10 selected pairs have `diffScore > 0`
(range 0.13–0.33, mean 0.22).

### 2.3 Selected Pairs

| # | More-complex Image | VC | Vis Type | Less-complex Image | VC | Vis Type | diffScore |
|---|---|:---:|---|---|:---:|---|:---:|
| 1 | `SciVisJ.259.7.png` | 0.75 | Cont.-ColorPatn | `SciVisJ.635.6.png` | 0.42 | Cont.-ColorPatn | 0.33 |
| 2 | `VisC.255.7.png` | 0.38 | Cont.-ColorPatn | `wsj108.png` | 0.55 | Line | 0.17 |
| 3 | `InfoVisJ.1933.13.png` | 0.68 | Grid | `SciVisJ.728.14.png` | 0.50 | Grid | 0.18 |
| 4 | `SciVisJ.867.6.png` | 0.59 | Line | `treasuryG07_2.png` | 0.31 | Line | 0.28 |
| 5 | `VisC.163.2.png` | 0.55 | Line | `whoK27_2.png` | 0.38 | Line | 0.17 |
| 6 | `VASTC.121.5.png` | 0.54 | Node-link | `InfoVisJ.485.9.png` | 0.67 | Point | 0.13 |
| 7 | `economist_daily_chart_152.png` | 0.55 | Point | `VisC.527.1.png` | 0.29 | Point | 0.26 |
| 8 | `whoQ32_2.png` | 0.56 | Point | `economist_daily_chart_393.png` | 0.42 | Point | 0.14 |
| 9 | `VASTJ.2012.4.png` | 0.77 | Text | `VASTJ.1763.8(2).png` | 0.47 | Text | 0.30 |
| 10 | `MorphableWordClouds21.png` | 0.58 | Text | `InfoVisJ.621.6(2).png` | 0.38 | Text | 0.20 |

*Note: In pair 2, `wsj108.png` (VC=0.55) has a higher VC than the
"more-complex" `VisC.255.7.png` (VC=0.38). The more/less-complex labels
follow the original pair-comparison direction in the study data; the
absolute VC values from the compiled ratings may not perfectly preserve
that ordering.*

### 2.4 Additions per Vis Type

| Vis Type | New Images Added | From Pairs |
|---|:---:|---|
| Cont.-ColorPatn | +3 | 1 (×2), 2 (×1) |
| Line | +5 | 2 (×1), 4 (×2), 5 (×2) |
| Grid | +2 | 3 (×2) |
| Node-link | +1 | 6 (×1) |
| Point | +5 | 6 (×1), 7 (×2), 8 (×2) |
| Text | +4 | 9 (×2), 10 (×2) |
| **Total** | **20** | 10 pairs |

---

## 3. Final 66-Image Set (33 Pairs)

### 3.1 Vis-Type Distribution (66 images, 63 non-anchor)

| Vis Type | Total (incl. anchor) | Test (excl. anchor) | Change from 46-set |
|---|:---:|:---:|:---:|
| Bar | 8 | 7 | — |
| Area | 7 | 7 | — |
| Glyph | 7 | 7 | — |
| Grid | 8 | 7 | +2 |
| Node-link | 7 | 7 | +1 |
| Cont.-ColorPatn | 7 | 7 | +3 |
| Point | 8 | 7 | +5 |
| Text | 7 | 7 | +4 |
| Line | 7 | 7 | +5 |
| **Total** | **66** | **63** | +20 |

All 9 vis-types now have exactly **7 test images**. The 3 anchor
exemplars (Bar, Grid, Point — one each) bring the total to 66.

### 3.2 Anchors (unchanged from 46-image set)

The same 3 anchor exemplars carry over unchanged:

| Image | Vis Type | VC |
|---|---|---|
| `VisC.503.6.png` | Bar | 0.22 |
| `InfoVisJ.1149.6(1).png` | Grid | 0.54 |
| `InfoVisJ.619.17.png` | Point | 0.95 |

### 3.3 GT Statistics

| Set | n (test) | GT mean | GT SD |
|---|:---:|:---:|:---:|
| 43-image pilot | 43 | — | ≈ 0.244 |
| 63-image balanced | 63 | TBD | TBD |

The pilot SD of ≈ 0.244 is higher than the full 510-image production set
(SD = 0.148) due to deliberate difficulty stratification. The 63-image
set inherits this stratification; statistics should be recomputed once
the expanded GT CSV is finalized.

---

## 4. Pending Artifacts

The 20 new images have been selected but not yet written to a formal
ground-truth CSV. The following artifacts need to be created:

- [ ] `Claude_vc_prediction/gt_all_66.csv` — expanded GT file combining
  `gt_all_46.csv` with the 20 new images and their `NormalizedVC` ratings
  from `phrase_reduction_v2/image_compiled_phrases.csv`.
- [ ] LLM scoring runs on the 20 new images for all relevant prompt
  variants (V0+TW, V1) to enable per-type analysis on the balanced set.
- [ ] Updated per-vis-type CCC table in `appendix_llm_evaluation.md` (§12.3
  or a new §13) using the 63-image test set.

---

## 5. Source Files

| File | Role |
|---|---|
| `Claude_vc_prediction/ResultsStepByStep - 0.postquestionare_all.csv` | Pair-comparison study results; source pool for pair selection |
| `phrase_reduction_v2/image_compiled_phrases.csv` | Canonical GT `NormalizedVC` ratings; eligibility filter |
| `Claude_vc_prediction/gt_all_46.csv` | Original 46-image GT file |
| `Claude_vc_prediction/gt_all_66.csv` | *(pending)* Expanded 66-image GT file |
