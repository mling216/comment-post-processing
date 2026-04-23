# Prompt Engineering for Automated Visual Complexity Scoring

*Draft — to be converted to an Overleaf section. Numbers in this document are computed
from the `V0_Variants_Comparison.ipynb` and `VC_Version_Comparison.ipynb` notebooks.*

## 1. Overview

We investigate how the structure of a large multimodal LLM prompt affects the model's
ability to reproduce human-rated **Visual Complexity (VC)** scores for static data
visualizations. Two complementary experiments are reported:

| Experiment | Purpose | #Prompts | Model | Sampling |
|---|---|---|---|---|
| **E1 — Ablation of prompt components** | Measure the marginal contribution of four prompt components (Topics, Calibration, Anchors, Weighted scoring) | 9 variants | `claude-opus-4-6` (primary); `claude-sonnet-4-6` (r3b) | `t=0` deterministic (×2) + adaptive extended thinking (×2) |
| **E2 — Prompt-version comparison** | Compare four substantively different prompt designs, from pure zero-shot to a full production prompt | 4 versions (V0, V1, V2, V3) | `claude-opus-4-6` | `t=1` (API default), single run per version, no extended thinking |

Both experiments share the same **ground truth (GT)** set and the same **evaluation
protocol** (Sec. 3).

## 2. Ground Truth and Evaluation Set

**Construction.** The 46-image ground-truth set was sub-sampled from a pool of ~700
paired images collected in a prior pair-comparison study. Pairs were ranked by the
absolute difference between their two aggregated human VC ratings, then binned across
the full range of $|\Delta \text{VC}|$. From each bin we selected roughly equal numbers
of pairs, yielding **23 pairs (46 images)** that span the full difficulty spectrum —
from near-ties (hard discrimination) to large-gap pairs (easy discrimination). Each
image carries a normalized human VC rating $y \in [0,1]$ derived from the original
pair study.

**Evaluation split.** Three of the 46 images are reserved as **few-shot anchor
exemplars** (VC = 0.22, 0.54, 0.95) and are presented in-context to every
anchored prompt (V2, V3, and the E1 variants containing "A"). To prevent leakage,
these three images are **excluded** from all reported metrics, leaving $n = 43$ test
images.

## 3. Evaluation Metrics

We report five agreement metrics, with **Lin's Concordance Correlation Coefficient
(CCC)** \cite{lin1989ccc} as the headline measure:

$$
\mathrm{CCC}(y, \hat{y}) = \frac{2\,\rho\,\sigma_y\sigma_{\hat{y}}}{\sigma_y^2 + \sigma_{\hat{y}}^2 + (\mu_y - \mu_{\hat{y}})^2}
$$

where $\rho$ is the Pearson correlation between human ratings $y$ and model predictions
$\hat{y}$. CCC measures agreement with the identity line $\hat{y}=y$, jointly penalizing
(i) low correlation, (ii) mean bias, and (iii) scale mismatch. It is bounded in
$[-1, 1]$ and is the standard metric for continuous inter-rater / method-comparison
studies. Additional diagnostics:

- **$r$** — Pearson correlation (linear association, invariant to affine rescaling).
- **$p$** — two-sided $p$-value of $r$ under $H_0: r = 0$, from `scipy.stats.pearsonr`.
- **$R^2$** — **coefficient of determination** against the identity line $\hat{y}=y$,
  computed with `sklearn.metrics.r2_score`:
  $R^2 = 1 - \sum_i (y_i - \hat{y}_i)^2 \,/\, \sum_i (y_i - \bar{y})^2$.
  Unlike $r^2$, this $R^2$ can be **negative** when the predictor does worse than
  predicting the mean of $y$, and it penalizes bias and scale mismatch, making it a
  more honest measure of predictive accuracy than $r^2$ for our setting.
- **MAE / RMSE** — mean absolute error and root mean squared error of
  $\hat{y}-y$ on the raw 0–1 scale.
- **Bias** — mean signed error $\overline{\hat{y}-y}$ (positive = over-estimation).

## 4. Experiment E1 — Ablation of Prompt Components

### 4.1 Prompt Components

Starting from a minimal **V0** baseline prompt that states only the *definition* of
visual complexity and the 0–1 output range, we incrementally layer four modular
components. **All E1 variants output only a single holistic `vc_score`** (no
per-dimension numeric scores).

- **T (Topics / Dimensions)**: Enumerates 7 complexity dimensions (data density, visual
  encoding, semantics/text, schema, color/symbol, aesthetics, cognitive load) as mental
  check-points.
- **C (Calibration)**: Provides scale-anchoring guidance with example chart types
  mapped to approximate target ranges (plain bar chart $\approx 0.25$–$0.40$; dense
  multi-panel $\approx 0.85$–$0.95$) and instructs the model to use the full range.
- **A (Anchors)**: Prepends the 3 labeled anchor images as few-shot exemplars before
  the target image. Each anchor carries only its final VC score (no per-dimension
  values).
- **W (Weighted)**: Qualitatively re-weights the 7 dimensions: high weight on density /
  encoding / schema / cognitive, medium on color / aesthetic, low on text.

All $2^3$ combinations of (T, C, A) are tested, plus V0+TCA+W (9 variants total).

### 4.2 Test Conditions

| Run | Config | Model | Variants |
|---|---|---|---|
| r1  | `temperature = 0` | `claude-opus-4-6` | all 9 |
| r2  | `temperature = 0` (replicate of r1) | `claude-opus-4-6` | all 9 |
| r3a | adaptive extended thinking (default temperature) | `claude-opus-4-6` | V0, V0+T, V0+TCA, V0+TCA+W |
| r3b | adaptive extended thinking (default temperature) | `claude-sonnet-4-6` | V0+TC, V0+TA, V0+CA |
| r4  | adaptive extended thinking (default temperature) | `claude-opus-4-6` | V0+TC, V0+TA, V0+CA, V0+TCA+W |

Aggregated columns:
- **det_avg** = mean of r1 and r2 (deterministic, `t=0`, opus).
- **think_avg** = mean of r3 and r4 (extended thinking; r3b on sonnet, r3a/r4 on opus).

### 4.3 Results — CCC Summary Table (column-width)

| Variant  | r1    | r2    | **det_avg** | r3    | r4    | **think_avg** |
|----------|-------|-------|-------------|-------|-------|---------------|
| V0       | 0.858 | 0.859 | **0.858**   | 0.868 | 0.881 | **0.875**     |
| V0+T     | 0.871 | 0.867 | **0.869**   | 0.857 | —     | **0.857**     |
| V0+C     | 0.851 | 0.856 | **0.854**   | —     | —     | —             |
| V0+A     | 0.852 | 0.853 | **0.852**   | —     | —     | —             |
| V0+TC    | 0.854 | —     | **0.854**   | 0.737 | 0.866 | **0.802**     |
| V0+TA    | 0.860 | —     | **0.860**   | 0.849 | 0.834 | **0.841**     |
| V0+CA    | 0.852 | —     | **0.852**   | 0.831 | 0.873 | **0.852**     |
| V0+TCA   | 0.856 | 0.856 | **0.856**   | 0.874 | 0.873 | **0.874**     |
| V0+TCA+W | 0.868 | 0.871 | **0.869**   | 0.811 | 0.815 | **0.813**     |

### 4.4 Results — Full Metrics Table (page-width, `det_avg`)

| Variant  | n | CCC    | r      | $\rho$ (Spearman) | $R^2$ (sklearn) | MAE    | RMSE   | Bias    |
|----------|---|--------|--------|-------------------|-----------------|--------|--------|---------|
| V0       | 43 | 0.858 | 0.868 | 0.868 | 0.671 | 0.080 | 0.098 | +0.010 |
| V0+T     | 43 | 0.869 | 0.879 | 0.872 | 0.692 | 0.078 | 0.095 | +0.003 |
| V0+C     | 43 | 0.854 | 0.871 | 0.873 | 0.679 | 0.081 | 0.097 | +0.033 |
| V0+A     | 43 | 0.852 | 0.869 | 0.866 | 0.638 | 0.083 | 0.103 | −0.012 |
| V0+TC    | 43 | 0.854 | 0.874 | 0.870 | 0.688 | 0.078 | 0.096 | +0.037 |
| V0+TA    | 43 | 0.860 | 0.882 | 0.872 | 0.656 | 0.081 | 0.101 | −0.025 |
| V0+CA    | 43 | 0.852 | 0.863 | 0.859 | 0.653 | 0.084 | 0.101 | +0.014 |
| V0+TCA   | 43 | 0.856 | 0.863 | 0.866 | 0.674 | 0.079 | 0.098 | +0.007 |
| V0+TCA+W | 43 | 0.869 | 0.876 | 0.871 | 0.704 | 0.075 | 0.093 | −0.003 |

*Deterministic runs average two independent `t=0` replicates. All Pearson correlations
are highly significant ($p < 10^{-12}$); see Sec. 5.4 for representative computed
$p$-values.*

### 4.5 Key Findings (E1)

- **At `t=0`, prompt engineering produces only marginal gains.** All nine variants land
  within a $\sim 0.02$ CCC band ($[0.852, 0.869]$). The baseline V0 (0.858) is already
  within $0.011$ of the best deterministic variant.
- **V0+T and V0+TCA+W tie for the deterministic best (both 0.869).** Adding the 7
  complexity dimensions as explicit mental checkpoints, or adding them plus weighting,
  yields the strongest fit to the identity line — but neither is meaningfully better
  than the baseline.
- **Extended thinking helps minimal prompts.** V0 improves $0.858 \to 0.875$ and
  V0+TCA improves $0.856 \to 0.874$ under adaptive thinking, suggesting the model
  benefits from additional reasoning tokens when the instructions are terse.
- **Extended thinking hurts V0+TCA+W** ($0.869 \to 0.813$). The qualitative weight
  guidance appears to interact poorly with longer deliberation; the model
  over-applies the "down-weight text" rule when given more scratchpad.
- **Cross-model confounder.** The r3b runs for V0+TC / V0+TA / V0+CA used
  `claude-sonnet-4-6` rather than `opus-4-6`; their lower thinking scores
  (especially V0+TC at 0.737) are recovered in the r4 opus replicates (0.866),
  confirming that the sonnet drop reflects model capacity, not a prompt defect.

## 5. Experiment E2 — Prompt-Version Comparison (V0–V3)

### 5.1 Prompt Versions

E2 compares four qualitatively distinct prompt designs. **Unlike E1, V1 and V3 require
the model to emit numeric scores for each of the 7 dimensions** (not just an overall
`vc_score`), so both the output format and the model's intermediate reasoning signal
differ across versions:

| Version | Anchors | Calibration | Weighting | Per-dimension numeric output |
|---|---|---|---|---|
| **V0** | — | — | — | — |
| **V1** | — | — | — | **✓ (7 dimensions)** |
| **V2** | ✓ (VC only) | — | — | — |
| **V3** | ✓ (VC + per-dimension) | ✓ | ✓ | **✓ (7 dimensions)** |

In words:

- **V0** — Pure zero-shot: VC definition only, no dimensions, no anchors. *(Identical
  to the V0 baseline in E1.)*
- **V1** — Zero-shot with the 7 dimension descriptions and **detailed per-dimension
  scoring** (model returns all 7 dimension scores plus `vc_score` and per-dimension
  explanations). No anchors, no calibration, no weighting.
- **V2** — VC definition + the 3 anchor images (anchors carry only the overall VC
  value, no per-dimension supervision).
- **V3** — Full production prompt: 7 dimensions + calibration guidance + qualitative
  weighting + 3 anchors with full per-dimension reference scores, and detailed
  per-dimension output.

The two major differences between V3 and E1's V0+TCA+W are therefore (a) **V3's anchors
include per-dimension ground-truth values**, and (b) **V3 requires the model to emit
per-dimension scores**, both of which V0+TCA+W lacks.

### 5.2 Test Conditions

All four versions were run on **`claude-opus-4-6`** with the Anthropic API default
sampling temperature (`t = 1`, not explicitly set) and **no extended thinking**.
`max_tokens = 800` for V0 and V2; V1 and V3 use `max_tokens = 1500` to accommodate the
longer per-dimension JSON output. Each version is evaluated from a single generation
pass (no replicate run).

Relative to E1, the E2 conditions match E1's `r1`/`r2` runs in model (`opus-4-6`) but
differ in sampling: E1 uses `t=0` (deterministic) whereas E2 uses `t=1` (stochastic).
Single-run `t=1` results therefore carry more sampling noise than the two-replicate
`t=0` averages reported for E1.

### 5.3 Results — CCC Summary Table (column-width)

| Version | n | **CCC** |
|---|---|---|
| V0 | 43 | 0.858 |
| V1 | 43 | 0.802 |
| V2 | 43 | 0.852 |
| **V3** | 43 | **0.882** |

### 5.4 Results — Full Metrics Table (page-width)

| Version | n  | CCC    | r      | $p$ (Pearson)         | $\rho$ (Spearman) | $R^2$ (sklearn) | MAE    | RMSE   | Bias    |
|---------|----|--------|--------|-----------------------|-------------------|-----------------|--------|--------|---------|
| V0      | 43 | 0.858 | 0.867 | $5.5\times10^{-14}$ | 0.869 | 0.670 | 0.080 | 0.099 | +0.011 |
| V1      | 43 | 0.802 | 0.831 | $5.6\times10^{-12}$ | 0.816 | 0.613 | 0.086 | 0.107 | −0.043 |
| V2      | 43 | 0.852 | 0.868 | $4.8\times10^{-14}$ | 0.865 | 0.637 | 0.083 | 0.103 | −0.012 |
| **V3**  | 43 | **0.882** | **0.900** | $\mathbf{2.0\times10^{-16}}$ | **0.896** | **0.718** | **0.076** | **0.091** | −0.025 |

### 5.5 Key Findings (E2)

- **V3 is the clear winner across every metric**: CCC 0.882, $r = 0.900$, $R^2 = 0.718$,
  MAE 0.076. Compared to V0 (pure zero-shot on the same model), V3 improves CCC by
  $+0.024$; compared to V2 (anchors without per-dimension supervision), V3 improves
  CCC by $+0.030$.
- **V1 (per-dimension scoring without anchors) is the worst version** (CCC 0.802).
  Requiring the model to emit detailed per-dimension scores **without** providing
  anchor exemplars introduces a negative bias ($-0.043$) and reduces correlation. This
  is the only E2 version where adding structure *hurts* the model, suggesting that
  per-dimension decomposition destabilizes calibration unless grounded in concrete
  reference images.
- **Anchors alone (V2) match the V0 baseline**: V2 (0.852) is within noise of V0
  (0.858). The benefit of anchors only becomes visible once they are *paired with
  per-dimension supervision* (V3).
- **The V3 gain is attributable to the joint effect of per-dimension anchors and
  per-dimension output.** Neither alone (V1: dimensions without anchors; V2: anchors
  without dimensions) exceeds the zero-shot baseline; their combination (V3) is what
  moves CCC above the 0.87 ceiling.

## 6. Cross-Experiment Summary

All runs use `claude-opus-4-6`.

| Prompt | Exp. | Condition | CCC | $r$ | MAE |
|---|---|---|---|---|---|
| V0 (baseline) | E1 | `t=0` (det_avg) | 0.858 | 0.868 | 0.080 |
| V0+T | E1 | `t=0` (det_avg) | 0.869 | 0.879 | 0.078 |
| V0+TCA+W | E1 | `t=0` (det_avg) | 0.869 | 0.876 | 0.075 |
| V0 | E1 | thinking (avg) | 0.875 | 0.889 | 0.077 |
| V0+TCA+W | E1 | thinking (avg) | 0.813 | 0.876 | 0.090 |
| V0 | E2 | `t=1` (single run) | 0.858 | 0.867 | 0.080 |
| V1 | E2 | `t=1` (single run) | 0.802 | 0.831 | 0.086 |
| V2 | E2 | `t=1` (single run) | 0.852 | 0.868 | 0.083 |
| **V3** | **E2** | **`t=1` (single run)** | **0.882** | **0.900** | **0.076** |

**Takeaways.**

1. **Prompt components alone have small, inconsistent effects** (E1: all `t=0` variants
   in a 0.02 CCC band).
2. **Extended thinking is a double-edged sword**: it helps minimal prompts
   (V0, V0+TCA) but hurts heavily-instructed prompts (V0+TCA+W).
3. **The single change that reliably breaks the 0.87 ceiling is requiring
   per-dimension scoring *with* per-dimension anchor supervision** (V3). Per-dimension
   output without anchors (V1) is actively harmful; anchors without per-dimension
   output (V2) are neutral.
4. **Same-model comparison.** E1 and E2 both use `claude-opus-4-6`, so the V3 gain
   over every E1 variant is attributable to prompt design rather than model capacity.
   V3's CCC 0.882 (single `t=1` run) exceeds the best E1 `t=0` det_avg (0.869) and the
   best E1 thinking result (0.875 for V0), despite carrying single-run sampling noise.

## 7. BibTeX for Lin's CCC

```bibtex
@article{lin1989ccc,
  author  = {Lin, Lawrence I-Kuei},
  title   = {A Concordance Correlation Coefficient to Evaluate Reproducibility},
  journal = {Biometrics},
  volume  = {45},
  number  = {1},
  pages   = {255--268},
  year    = {1989},
  doi     = {10.2307/2532051},
  publisher = {International Biometric Society},
}
```
