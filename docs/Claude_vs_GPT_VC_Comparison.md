# Claude Opus 4.6 vs GPT 5.4 — Visual Complexity Pairwise Comparison

**Date:** April 2026  
**Task:** Determine which image in each of 23 pairs is more visually complex  
**Method:** Taxonomy-based evaluation (7 topics / 19 subtopics)  
**Ground truth:** Human-rated VC scores from `image_pairs_by_vc_diff.xlsx`

---

## 1. Overall Accuracy vs Ground Truth

Pair 1 is excluded from accuracy calculations because both images have identical human VC scores (0.57 = 0.57), making the ground truth a **tie** — neither model can be correct or incorrect.

| Metric | Claude Opus 4.6 | GPT 5.4 |
|--------|:-:|:-:|
| **Correct / 22** | **20** | **18** |
| **Accuracy** | **90.9%** | **81.8%** |
| Both correct | 18 | 18 |
| Only this model correct | 2 (Pairs 2, 14) | 0 |
| Both wrong | 2 (Pairs 3, 5) | 2 (Pairs 3, 5) |

Claude is strictly more accurate: every pair GPT got right, Claude also got right. Claude additionally got Pairs 2 and 14 correct where GPT did not.

### Effect of Including Pair 1 (Tie)

If Pair 1 is included (n = 23) and we award credit for being "closer to a tie," Claude's near-parity subtopic split (9–7, margin = 2) arguably deserves partial or full credit, while GPT's one-sided reasoning does not. The table below shows accuracy under different scoring rules:

| Scoring Rule | Claude | GPT |
|--------------|:------:|:---:|
| Exclude Pair 1 (baseline, n = 22) | 20/22 = **90.9%** | 18/22 = **81.8%** |
| Include Pair 1, neither gets credit (n = 23) | 20/23 = **87.0%** | 18/23 = **78.3%** |
| Include Pair 1, credit for closer-to-tie (n = 23) | 21/23 = **91.3%** | 18/23 = **78.3%** |

Under all three rules Claude leads by 9–13 percentage points. Crediting the tie further widens the gap from 9.1 pp to 13.0 pp.

---

## 2. Per-Pair Detail

| Pair | VC₁ | VC₂ | Diff | Ground Truth | Claude | GPT | Claude | GPT | Agree? |
|:----:|:---:|:---:|:----:|:---:|:---:|:---:|:---:|:---:|:---:|
| | | | | | *prediction* | *prediction* | *correct?* | *correct?* | |
| 1 | 0.57 | 0.57 | 0.00 | Tie | Image 2 | Image 1 | ~ | ~ | N |
| 2 | 0.57 | 0.54 | 0.03 | Image 1 | Image 1 | Image 2 | **Y** | N | N |
| 3 | 0.67 | 0.72 | 0.05 | Image 2 | Image 1 | Image 1 | N | N | Y |
| 4 | 0.76 | 0.83 | 0.07 | Image 2 | Image 2 | Image 2 | **Y** | **Y** | Y |
| 5 | 0.62 | 0.53 | 0.09 | Image 1 | Image 2 | Image 2 | N | N | Y |
| 6 | 0.69 | 0.57 | 0.12 | Image 1 | Image 1 | Image 1 | **Y** | **Y** | Y |
| 7 | 0.77 | 0.63 | 0.14 | Image 1 | Image 1 | Image 1 | **Y** | **Y** | Y |
| 8 | 0.70 | 0.53 | 0.17 | Image 1 | Image 1 | Image 1 | **Y** | **Y** | Y |
| 9 | 0.54 | 0.35 | 0.19 | Image 1 | Image 1 | Image 1 | **Y** | **Y** | Y |
| 10 | 0.67 | 0.88 | 0.21 | Image 2 | Image 2 | Image 2 | **Y** | **Y** | Y |
| 11 | 0.52 | 0.30 | 0.22 | Image 1 | Image 1 | Image 1 | **Y** | **Y** | Y |
| 12 | 0.74 | 0.50 | 0.24 | Image 1 | Image 1 | Image 1 | **Y** | **Y** | Y |
| 13 | 0.58 | 0.31 | 0.27 | Image 1 | Image 1 | Image 1 | **Y** | **Y** | Y |
| 14 | 0.77 | 0.48 | 0.29 | Image 1 | Image 1 | Image 2 | **Y** | N | N |
| 15 | 0.84 | 0.52 | 0.32 | Image 1 | Image 1 | Image 1 | **Y** | **Y** | Y |
| 16 | 0.77 | 0.43 | 0.34 | Image 1 | Image 1 | Image 1 | **Y** | **Y** | Y |
| 17 | 0.73 | 0.36 | 0.37 | Image 1 | Image 1 | Image 1 | **Y** | **Y** | Y |
| 18 | 0.75 | 0.36 | 0.39 | Image 1 | Image 1 | Image 1 | **Y** | **Y** | Y |
| 19 | 0.78 | 0.36 | 0.42 | Image 1 | Image 1 | Image 1 | **Y** | **Y** | Y |
| 20 | 0.88 | 0.45 | 0.43 | Image 1 | Image 1 | Image 1 | **Y** | **Y** | Y |
| 21 | 0.73 | 0.28 | 0.45 | Image 1 | Image 1 | Image 1 | **Y** | **Y** | Y |
| 22 | 0.92 | 0.38 | 0.54 | Image 1 | Image 1 | Image 1 | **Y** | **Y** | Y |
| 23 | 0.95 | 0.36 | 0.59 | Image 1 | Image 1 | Image 1 | **Y** | **Y** | Y |

---

## 3. Inter-Model Agreement

| Metric | Value |
|--------|-------|
| Same verdict | 20 / 23 (87.0%) |
| Disagreements | 3 / 23 (13.0%) |

### Disagreement Breakdown

| Pair | Claude | GPT | Ground Truth | VC Diff | Who was right? |
|:----:|:------:|:---:|:-----------:|:-------:|:--------------:|
| 1 | Image 2 | Image 1 | Tie (0.57 = 0.57) | 0.00 | Neither (tie) |
| 2 | Image 1 | Image 2 | Image 1 (0.57 > 0.54) | 0.03 | **Claude** |
| 14 | Image 1 | Image 2 | Image 1 (0.77 > 0.48) | 0.29 | **Claude** |

In all 3 disagreements, Claude was correct (or the pair was a tie). GPT never disagreed with Claude and was correct.

---

## 4. Accuracy by VC Difference Magnitude

Pairs are grouped by the absolute difference between the two human VC scores. Larger differences should be easier to judge.

| Difficulty Bucket | n | Claude | GPT |
|-------------------|:-:|:------:|:---:|
| Tiny (0.00–0.09) | 4* | 2/4 (50%) | 1/4 (25%) |
| Small (0.10–0.19) | 4 | 4/4 (100%) | 4/4 (100%) |
| Medium (0.20–0.39) | 9 | 9/9 (100%) | 8/9 (89%) |
| Large (0.40+) | 5 | 5/5 (100%) | 5/5 (100%) |

*\*Excludes Pair 1 (tie). The 4 "tiny" pairs are 2–5 (diff = 0.03–0.09).*

**Key finding:** Both models are perfect when VC diff ≥ 0.10. The hard cases are pairs with diff < 0.10, where Claude achieves 50% and GPT only 25%. GPT's extra miss at diff = 0.29 (Pair 14) is notable — a medium-difficulty pair that Claude handled correctly.

---

## 5. Error Analysis

### Pair 1 — Intentional Tie (VC₁ = VC₂ = 0.57)

Neither model predicted "Tie," but Claude's result is **closer to a tie**:

| Metric | Claude (picked Image 2) | GPT (picked Image 1) |
|--------|:-:|:-:|
| Subtopic wins for Image 1 | 7 | N/A (no subtopic data) |
| Subtopic wins for Image 2 | 9 | N/A |
| Ties | 3 | N/A |
| Margin | **2 subtopics** (near-parity) | Presented as one-sided |
| Dimensions cited for winner | Text, color, annotations | Element quantity, density, repetition, processing |
| Dimensions cited for loser | Abstraction, ambiguity, domain | *(none — no counter-argument)* |

Claude's 9–7 split with 3 ties reflects near-parity and its rationale acknowledges strengths on both sides. GPT's reasoning cites 4 dimensions favoring Image 1 with zero dimensions favoring Image 2, framing it as a clear-cut win. Claude better captures the ground-truth tie.

### Pairs Both Got Wrong

| Pair | Images | GT | Both said | Why hard? |
|------|--------|----|-----------|-----------|
| 3 | VASTJ.2012.9(2) vs SciVisJ.1236.10 | Image 2 (0.72 > 0.67) | Image 1 | GT says the small-multiple heatmaps (0.72) are slightly more complex than the ThemeRiver+word-clouds (0.67), but both models were swayed by Image 1's heterogeneous forms. VC diff is only 0.05. |
| 5 | v483_n7391_8_f5 vs InfoVisJ.1236.13 | Image 1 (0.62 > 0.53) | Image 2 | GT says the physics contour plot (0.62) beats the Netherlands cartogram (0.53), but both models favored Image 2's multi-hue color variety. VC diff is 0.09. |

Both shared errors involve small VC differences (≤ 0.09) where the human-rated "more complex" image has a less immediately striking visual presentation but higher domain complexity.

### Pairs Only GPT Got Wrong

| Pair | Images | GT | GPT said | Claude said | Why GPT missed? |
|------|--------|----|----------|-------------|-----------------|
| 2 | VisJ.1949.18(3) vs InfoVisJ.1558.9 | Image 1 (0.57 > 0.54) | Image 2 | Image 1 | GPT weighted annotation load and linking guides in Image 2 over 3D abstraction. Claude correctly prioritized 3D glyph rendering + domain barrier + zero labels. VC diff = 0.03. |
| 14 | VisJ.1541.1(2) vs InfoVisC.133.5(3) | Image 1 (0.77 > 0.48) | Image 2 | Image 1 | GPT saw the parallel-sets crossing ribbons as more complex than the FEM mesh. Claude correctly recognized thousands of triangular facets + domain-specific encoding. VC diff = 0.29 — a clear margin GPT missed. |

---

## 6. Reasoning Style Comparison

### Quantitative Overview

| Metric | Claude Opus 4.6 | GPT 5.4 |
|--------|:-:|:-:|
| Avg. word count per pair | 46.4 | 44.2 |
| Min / Max words | 29 / 66 | 32 / 51 |
| Avg. sentences per pair | 3.2 | 1.0 |
| Sentence structure | Multi-sentence, comparative | Single compound sentence |

### Structural Differences

- **Claude** writes 2–5 sentences per pair: typically (1) describes Image 1, (2) describes Image 2, (3) states who wins and on which dimensions. Often includes specific counts (e.g., "~300+ nodes," "~3500+ data cells," "7 coordinated views labeled A–G").
  
- **GPT** writes exactly 1 long compound sentence per pair using a consistent template: *"Image X is higher on [subtopic list] because [description] whereas [other image description]."* 

### Subtopic Keyword Frequency in Reasoning

How often each model's rationale text explicitly references taxonomy-related concepts:

| Concept | Claude (of 23) | GPT (of 23) |
|---------|:-:|:-:|
| Color / hue / palette | **19** | 15 |
| Annotations / labels / legend | 16 | **18** |
| Visual clutter / density | 11 | **12** |
| Domain familiarity | **8** | 3 |
| Symbols / texture / glyph | **8** | 5 |
| Position / scale / layout | 7 | **9** |
| Encoding interpretability | **6** | 3 |
| Dimensionality / 3D | 6 | **11** |
| Abstraction | 4 | **5** |
| Processing effort / cognitive | 0 | **19** |
| Semantic clarity | 1 | **9** |
| Information volume | 0 | **11** |
| Element quantity | 2 | **12** |
| Text volume | 1 | **5** |
| Interpretive difficulty | 1 | **5** |

**Notable patterns:**
- GPT mentions "processing effort" in 19/23 rationales — it appears to be a default closing phrase in GPT's template.
- Claude more frequently references **color** (19 vs 15), **domain familiarity** (8 vs 3), and **encoding interpretability** (6 vs 3).
- GPT more frequently references **element quantity** (12 vs 2), **information volume** (11 vs 0), and **dimensionality** (11 vs 6).
- Claude's emphasis on domain-specific knowledge and color encoding may explain its better performance on Pair 2 (3D glyph abstraction) and Pair 14 (FEM mesh domain complexity).

---

## 7. Claude Subtopic Score Distribution (19 subtopics × 23 pairs)

Claude provided per-subtopic verdicts (Image 1 / Image 2 / Tie) for all 23 pairs. GPT did not provide subtopic-level scores, only overall verdicts.

### By Subtopic

| # | SubTopic | Image 1 | Image 2 | Tie |
|:-:|----------|:-------:|:-------:|:---:|
| 1.1 | Information Volume | 19 | 2 | 2 |
| 1.2 | Element Quantity | 16 | 4 | 3 |
| 1.3 | Visual Clutter & Overlap | 19 | 4 | 0 |
| 2.1 | Graphical Forms & Primitives | 19 | 4 | 0 |
| 2.2 | Position, Scale & Organization | 16 | 3 | 4 |
| 2.3 | Encoding Interpretability | 21 | 1 | 1 |
| 3.1 | Annotations & Labels | 10 | 11 | 2 |
| 3.2 | Text Volume & Content | 10 | 11 | 2 |
| 3.3 | Typography & Readability | 9 | 10 | 4 |
| 4.1 | Domain Familiarity | 20 | 2 | 1 |
| 4.2 | Dimensionality & Structure | 6 | 0 | 17 |
| 4.3 | Abstraction Level | 16 | 2 | 5 |
| 5.1 | Color Palette & Contrast | 18 | 5 | 0 |
| 5.2 | Symbols & Texture | 12 | 2 | 9 |
| 6.1 | Visual Disorganization | 18 | 4 | 1 |
| 6.2 | Perceptual Ambiguity | 19 | 2 | 2 |
| 7.1 | Interpretive Difficulty | 22 | 1 | 0 |
| 7.2 | Semantic Clarity | 20 | 3 | 0 |
| 7.3 | Processing Time & Effort | 20 | 3 | 0 |

### By Topic (aggregated)

| Topic | Image 1 Win % | Image 2 Win % | Tie % |
|-------|:---:|:---:|:---:|
| Data Density / Image Clutter | 78.3% | 14.5% | 7.2% |
| Visual Encoding Clarity | 81.2% | 11.6% | 7.2% |
| Semantics / Text Legibility | 42.0% | **46.4%** | 11.6% |
| Schema | 60.9% | 5.8% | 33.3% |
| Color, Symbol & Texture | 65.2% | 15.2% | 19.6% |
| Aesthetics Uncertainty | 80.4% | 13.0% | 6.5% |
| Immediacy / Cognitive Load | 89.9% | 10.1% | 0.0% |

**Key insight:** Semantics/Text is the only topic where Image 2 wins more often than Image 1. This is because the "more complex" Image 1 in most pairs is a dense scientific/abstract visualization with fewer labels, while Image 2 tends to be a well-annotated conventional chart. The simpler image often has more text.

---

## 8. Summary

| Dimension | Claude Opus 4.6 | GPT 5.4 |
|-----------|:-:|:-:|
| Accuracy (22 non-tie pairs) | **90.9%** (20/22) | 81.8% (18/22) |
| Hard pairs (diff < 0.10) | **50%** (2/4) | 25% (1/4) |
| Easy pairs (diff ≥ 0.10) | **100%** (18/18) | 94.4% (17/18) |
| Inter-model agreement | 87.0% (20/23) | — |
| Unique errors | 0 | 2 (Pairs 2, 14) |
| Shared errors | 2 (Pairs 3, 5) | 2 (Pairs 3, 5) |
| Reasoning length | 46 words / 3.2 sentences | 44 words / 1.0 sentence |
| Subtopic-level detail | Full 19-subtopic breakdown | Overall verdict only |
| Reasoning emphasis | Color, domain, encoding | Processing effort, element quantity |

Both models perform well on clear-cut pairs (VC diff ≥ 0.10) but struggle with near-tie pairs. Claude's additional attention to domain familiarity and encoding complexity appears to give it an edge on ambiguous cases.
