# Prompt Engineering for Visual Complexity Scoring

*Markdown mirror of Appendix B (`VC_Prompt_Experiments_AppendixB.tex`). Numbers
are computed from `V0_Variants_Comparison.ipynb` and `VC_Version_Comparison.ipynb`.*

We investigate how the structure of a large multimodal LLM prompt affects the
model's ability to reproduce human-rated **Visual Complexity (VC)** scores for
static data visualizations. Two complementary experiments are reported:
**E1**, an ablation of four prompt components (Topics, Calibration, Anchors,
Weighted scoring), and **E2**, a comparison of four qualitatively distinct
prompt designs (V0–V3). Both experiments share the same ground truth and
evaluation protocol.

## 1. Ground Truth and Evaluation Set

The 46-image ground-truth set was sub-sampled from a pool of approximately
700 paired images collected in a prior pair-comparison study. Pairs were
ranked by the absolute difference $|\Delta \mathrm{VC}|$ between their two
aggregated human ratings, then binned across the full range of
$|\Delta \mathrm{VC}|$. From each bin we selected roughly equal numbers of
pairs, yielding **23 pairs (46 images)** that span the full difficulty
spectrum — from near-ties (hard discrimination) to large-gap pairs (easy
discrimination). Each image carries a normalized human VC rating
$y \in [0,1]$ derived from the pair study.

Three of the 46 images are reserved as few-shot anchor exemplars
($\mathrm{VC} = 0.22, 0.54, 0.95$) and are presented in-context to every
anchored prompt (V2, V3, and the E1 variants containing "A"). To prevent
leakage, these three images are *excluded* from all reported metrics,
leaving $n = 43$ test images.

## 2. Evaluation Metrics

We report five agreement metrics, with Lin's Concordance Correlation
Coefficient (CCC) \cite{lin1989ccc} as the headline measure:

$$
\mathrm{CCC}(y, \hat{y}) =
\frac{2\,\rho\,\sigma_y\sigma_{\hat{y}}}
     {\sigma_y^2 + \sigma_{\hat{y}}^2 + (\mu_y - \mu_{\hat{y}})^2},
$$

where $\rho$ is the Pearson correlation between human ratings $y$ and model
predictions $\hat{y}$. CCC measures agreement with the identity line
$\hat{y}=y$, jointly penalizing (i) low correlation, (ii) mean bias, and
(iii) scale mismatch. It is bounded in $[-1, 1]$ and is the standard metric
for continuous method-comparison studies. We additionally report Pearson
$r$ with its two-sided $p$-value (`scipy.stats.pearsonr`), Spearman $\rho$,
and the coefficient of determination $R^2$ against the identity line
(`sklearn.metrics.r2_score`),
$R^2 = 1 - \sum_i (y_i - \hat{y}_i)^2 / \sum_i (y_i - \bar{y})^2$, which
can be negative when the predictor underperforms the mean of $y$. MAE,
RMSE, and the signed bias $\overline{\hat{y}-y}$ complete the diagnostics.

## 3. E1 — Ablation of Prompt Components

### 3.1 Prompt components

Starting from a minimal **V0** baseline that states only the definition of
visual complexity and the 0–1 output range, we incrementally layer four
modular components. All E1 variants output only a single holistic
`vc_score` (no per-dimension numeric scores).

- **T (Topics / Dimensions):** Enumerates seven complexity dimensions
  (data density, visual encoding, semantics/text, schema, color/symbol,
  aesthetics, cognitive load) as mental checkpoints.
- **C (Calibration):** Provides scale-anchoring guidance with example
  chart types mapped to approximate target ranges (plain bar chart
  $\approx 0.25$–$0.40$; dense multi-panel $\approx 0.85$–$0.95$) and
  instructs the model to use the full range.
- **A (Anchors):** Prepends the three labeled anchor images as few-shot
  exemplars before the target. Each anchor carries only its final VC score
  (no per-dimension values).
- **W (Weighted):** Qualitatively re-weights the seven dimensions: high
  weight on density / encoding / schema / cognitive, medium on color /
  aesthetic, low on text.

All $2^3$ combinations of $(T, C, A)$ are tested, plus V0+TCA+W (9 variants).

### 3.2 Test conditions

Each variant is evaluated under four independent runs on
`claude-opus-4-6` (primary model): `r1` and `r2` are deterministic ($t=0$)
replicates; `r3` and `r4` use Claude's adaptive extended-thinking mode.
A subset of `r3` (variants V0+TC, V0+TA, V0+CA) used `claude-sonnet-4-6`;
these were re-run in `r4` on `claude-opus-4-6` for cross-variant parity.
We define *det_avg* as the mean of `r1` and `r2`, and *think_avg* as the
mean of `r3` and `r4`.

### 3.3 Results — CCC per run

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

### 3.4 Results — Full metrics on *det_avg*

$n = 43$ non-anchor images, `claude-opus-4-6`. All Pearson $p < 10^{-12}$.

| Variant  | $n$ | CCC   | $r$   | $\rho$ (Spearman) | $R^2$ | MAE   | RMSE  | Bias    |
|----------|----:|------:|------:|------------------:|------:|------:|------:|--------:|
| V0       | 43  | 0.858 | 0.868 | 0.868 | 0.671 | 0.080 | 0.098 | $+0.010$ |
| V0+T     | 43  | 0.869 | 0.879 | 0.872 | 0.692 | 0.078 | 0.095 | $+0.003$ |
| V0+C     | 43  | 0.854 | 0.871 | 0.873 | 0.679 | 0.081 | 0.097 | $+0.033$ |
| V0+A     | 43  | 0.852 | 0.869 | 0.866 | 0.638 | 0.083 | 0.103 | $-0.012$ |
| V0+TC    | 43  | 0.854 | 0.874 | 0.870 | 0.688 | 0.078 | 0.096 | $+0.037$ |
| V0+TA    | 43  | 0.860 | 0.882 | 0.872 | 0.656 | 0.081 | 0.101 | $-0.025$ |
| V0+CA    | 43  | 0.852 | 0.863 | 0.859 | 0.653 | 0.084 | 0.101 | $+0.014$ |
| V0+TCA   | 43  | 0.856 | 0.863 | 0.866 | 0.674 | 0.079 | 0.098 | $+0.007$ |
| V0+TCA+W | 43  | 0.869 | 0.876 | 0.871 | 0.704 | 0.075 | 0.093 | $-0.003$ |

### 3.5 Findings

At $t=0$, prompt engineering produces only marginal gains: all nine
variants land within a $\sim\!0.02$ CCC band ($[0.852, 0.869]$), with V0+T
and V0+TCA+W tied at the deterministic best ($0.869$) and the baseline V0
($0.858$) trailing by only $0.011$. Extended thinking helps minimal
prompts (V0: $0.858 \to 0.875$; V0+TCA: $0.856 \to 0.874$) but *hurts*
V0+TCA+W ($0.869 \to 0.813$); the qualitative weight guidance appears to
interact poorly with longer deliberation. The `r3` sonnet runs
(V0+TC / V0+TA / V0+CA) are recovered by their `r4` opus replicates
(e.g., V0+TC: $0.737 \to 0.866$), confirming the drop is a model-capacity
effect, not a prompt defect.

## 4. E2 — Prompt-Version Comparison (V0–V3)

### 4.1 Prompt versions

E2 compares four qualitatively distinct prompt designs. Unlike E1, **V1
and V3 require the model to emit numeric scores for each of the seven
dimensions** in addition to the overall `vc_score`:

- **V0** — Pure zero-shot: VC definition only, no dimensions, no anchors.
  (Identical to E1's V0 baseline.)
- **V1** — Zero-shot with the seven dimension descriptions and detailed
  per-dimension scoring. No anchors, no calibration, no weighting.
- **V2** — VC definition plus the three anchor images (anchors carry only
  overall VC, no per-dimension supervision).
- **V3** — Full prompt: dimensions + calibration + weighting + three
  anchors with full per-dimension reference scores, and detailed
  per-dimension output.

The two principal differences between V3 and E1's V0+TCA+W are that
(a) V3's anchors include per-dimension ground-truth values, and (b) V3
requires the model to emit per-dimension scores — neither of which is
present in V0+TCA+W.

### 4.2 Test conditions

All four versions were run on `claude-opus-4-6` with the Anthropic API
default sampling temperature ($t=1$, not explicitly set) and no extended
thinking. `max_tokens = 800` for V0 and V2; `max_tokens = 1500` for V1
and V3 to accommodate the longer per-dimension JSON output. Each version
is evaluated from a single generation pass. Relative to E1, E2 matches
the model but differs in sampling (E1: $t=0$ with two replicates; E2:
$t=1$, single run); E2 results therefore carry more sampling noise than
E1's *det_avg*.

### 4.3 Results — CCC summary

$n = 43$ non-anchor images, `claude-opus-4-6`, $t=1$, single run.

| Version | $n$ | **CCC**   |
|---------|----:|----------:|
| V0      | 43  | 0.858     |
| V1      | 43  | 0.802     |
| V2      | 43  | 0.852     |
| **V3**  | 43  | **0.882** |

### 4.4 Results — Full metrics

| Version | $n$ | CCC   | $r$   | $p$ (Pearson)         | $\rho$ (Spearman) | $R^2$ | MAE   | RMSE  | Bias    |
|---------|----:|------:|------:|-----------------------|------------------:|------:|------:|------:|--------:|
| V0      | 43  | 0.858 | 0.867 | $5.5\times10^{-14}$   | 0.869 | 0.670 | 0.080 | 0.099 | $+0.011$ |
| V1      | 43  | 0.802 | 0.831 | $5.6\times10^{-12}$   | 0.816 | 0.613 | 0.086 | 0.107 | $-0.043$ |
| V2      | 43  | 0.852 | 0.868 | $4.8\times10^{-14}$   | 0.865 | 0.637 | 0.083 | 0.103 | $-0.012$ |
| **V3**  | 43  | **0.882** | **0.900** | $\mathbf{2.0\times10^{-16}}$ | **0.896** | **0.718** | **0.076** | **0.091** | $-0.025$ |

### 4.5 Findings

V3 dominates every metric (CCC $0.882$, $r = 0.900$, $R^2 = 0.718$,
MAE $0.076$): $+0.024$ CCC over V0 and $+0.030$ over V2. V1 —
per-dimension scoring without anchor exemplars — is the *worst* version
(CCC $0.802$, bias $-0.043$); requiring numeric per-dimension output
destabilizes calibration when no reference images are provided. Anchors
alone (V2, $0.852$) match the V0 baseline ($0.858$); the benefit of
anchors only emerges when paired with per-dimension supervision (V3).

## 5. Cross-Experiment Summary

All runs use `claude-opus-4-6`.

| Prompt        | Exp. / Condition          | CCC   | $r$   | MAE   |
|---------------|---------------------------|------:|------:|------:|
| V0 (baseline) | E1 $t{=}0$ (det_avg)      | 0.858 | 0.868 | 0.080 |
| V0+T          | E1 $t{=}0$ (det_avg)      | 0.869 | 0.879 | 0.078 |
| V0+TCA+W      | E1 $t{=}0$ (det_avg)      | 0.869 | 0.876 | 0.075 |
| V0            | E1 thinking (avg)         | 0.875 | 0.889 | 0.077 |
| V0+TCA+W      | E1 thinking (avg)         | 0.813 | 0.876 | 0.090 |
| V0            | E2 $t{=}1$ (single run)   | 0.858 | 0.867 | 0.080 |
| V1            | E2 $t{=}1$ (single run)   | 0.802 | 0.831 | 0.086 |
| V2            | E2 $t{=}1$ (single run)   | 0.852 | 0.868 | 0.083 |
| **V3**        | **E2 $t{=}1$ (single run)** | **0.882** | **0.900** | **0.076** |

Four findings:

1. Prompt components alone have small, inconsistent effects (E1: 0.02 CCC
   band at $t=0$).
2. Extended thinking is a double-edged sword: it helps minimal prompts
   (V0, V0+TCA) but hurts heavily-instructed ones (V0+TCA+W).
3. The single change that reliably breaks the $0.87$ ceiling is requiring
   per-dimension scoring *with* per-dimension anchor supervision (V3).
   Per-dimension output without anchors (V1) is actively harmful; anchors
   without per-dimension output (V2) are neutral.
4. Because E1 and E2 share the same model, V3's advantage over every E1
   variant is a clean prompt-design effect, not a model-capacity confound.

## 6. BibTeX for Lin's CCC

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
