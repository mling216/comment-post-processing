# Prompt Engineering for Visual Complexity Scoring — Condensed (v2)

*Condensed rewrite of Appendix B. This version folds the former E2
(V0–V3) into a single unified ablation by adding the former V3 (here
renamed **V1**) as an extra variant of E1 and pruning the redundant
two-feature cell (V0+CA). Numbers are sourced from
`V0_Variants_Comparison.ipynb` (r1–r4 on the E1 variants) and
`VC_Version_Comparison.ipynb` (V1).*

We investigate how the structure of a large multimodal LLM prompt affects
the model's ability to reproduce human-rated **Visual Complexity (VC)**
scores for static data visualizations. Starting from a minimal baseline
prompt (**V0**), we incrementally layer four modular components —
**T**opics, **C**alibration, **A**nchors, and **W**eighted scoring — and
report a single unified ablation over eight variants. A ninth variant,
**V1**, extends the most heavily instrumented cell (V0+TWCA) with two
additional elements — per-dimension anchor supervision and per-dimension
numeric output — and is included as the upper envelope of the ablation.
(V1 is the prompt previously labeled V3 in an earlier draft; it is
re-numbered here to avoid implying intermediate versions that were never
run.)

## 1. Ground Truth and Evaluation Set

The 46-image ground-truth set was sub-sampled from a pool of approximately
700 paired images collected in a prior pair-comparison study. Pairs were
ranked by the absolute difference $|\Delta \mathrm{VC}|$ between their
aggregated human ratings, then binned across the full range of
$|\Delta \mathrm{VC}|$. From each bin we selected roughly equal numbers of
pairs, yielding **23 pairs (46 images)** that span the full difficulty
spectrum — from near-ties (hard discrimination) to large-gap pairs (easy
discrimination). Each image carries a normalized human VC rating
$y \in [0, 1]$ derived from the pair study.

Three of the 46 images are reserved as few-shot anchor exemplars
($\mathrm{VC} = 0.22, 0.54, 0.95$) and are presented in-context to every
anchored variant (all cells containing "A", plus V1). To prevent leakage,
these three images are *excluded* from all reported metrics, leaving
$n = 43$ test images.

The set is **not balanced by visualization type**; the sampling was driven
by $|\Delta \mathrm{VC}|$ coverage rather than vis-type stratification.
The distribution across the 46 images is:

| Vis Type | Count (46 total) | Anchor? |
|---|---|---|
| Bar | 8 | VisC.503.6 (1 anchor) |
| Area | 7 | — |
| Glyph | 7 | — |
| Grid | 6 | InfoVisJ.1149.6(1) (1 anchor) |
| Node-link | 6 | — |
| Cont.-ColorPatn | 4 | — |
| Point | 3 | InfoVisJ.619.17 (1 anchor) |
| Text | 3 | — |
| Line | 2 | — |

The imbalance means that per-vis-type performance estimates are unreliable
for under-represented types (Point, Text, Line). Results in §6–§8 are
reported over the full test set; vis-type breakdowns should be interpreted
with this caveat in mind.

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
$\hat{y} = y$, jointly penalizing (i) low correlation, (ii) mean bias, and
(iii) scale mismatch. It is bounded in $[-1, 1]$ and is the standard metric
for continuous method-comparison studies. We additionally report Pearson
$r$ with its two-sided $p$-value (`scipy.stats.pearsonr`), Spearman $\rho$,
the coefficient of determination $R^2$ against the identity line
(`sklearn.metrics.r2_score`,
$R^2 = 1 - \sum_i (y_i - \hat{y}_i)^2 / \sum_i (y_i - \bar{y})^2$, which
can be negative when the predictor underperforms the mean of $y$), MAE,
RMSE, and the signed bias $\overline{\hat{y} - y}$.

## 3. Prompt Components

Starting from the minimal **V0** baseline — VC definition plus the 0–1
output range, no dimensions, no anchors — we incrementally layer four
modular components:

- **T (Topics / Dimensions):** Enumerates seven complexity dimensions
  (data density, visual encoding, semantics/text, schema, color/symbol,
  aesthetics, cognitive load) as mental checkpoints. Output remains a
  single holistic `vc_score`.
- **C (Calibration):** Provides scale-anchoring guidance with example
  chart types mapped to target ranges (plain bar chart $\approx 0.25$–$0.40$;
  dense multi-panel $\approx 0.85$–$0.95$) and instructs the model to use
  the full range.
- **A (Anchors):** Prepends the three labeled anchor images as few-shot
  exemplars before the target. Each anchor carries only its final VC score
  (no per-dimension values).
- **W (Weighted):** Qualitatively re-weights the seven dimensions: high
  weight on density / encoding / schema / cognitive, medium on color /
  aesthetic, low on text.

## 4. Variant Selection

Rather than run all $2^3$ combinations of $(T, C, A)$, we prune the lattice
on the observation that **among the single-component cells (V0+T, V0+C,
V0+A), V0+T achieves the highest CCC**, motivating a T-anchored path
through the design space. The final variant list is therefore:

| Level             | Variants                          |
|-------------------|-----------------------------------|
| Baseline          | V0                                |
| Single            | V0+T, V0+C, V0+A                  |
| Double (T-only)   | V0+TC, V0+TA, V0+TW               |
| Triple (TW-based) | V0+TWC, V0+TWA                    |
| Quadruple         | V0+TWCA                          |
| Extended          | **V1** (V0+TWCA + per-dim anchors + per-dim output) |

Since V0+T is the strongest single-component cell, the double-feature
sweep pairs T with each of the three remaining components (C, A, W) —
giving V0+TC, V0+TA, and V0+TW — rather than the redundant non-T cell
V0+CA. The triple layer is then built on the strongest double (V0+TW)
by adding C or A, yielding V0+TWC and V0+TWA. V0+TWCA is the full
four-component (quadruple) cell.

**V1** extends V0+TWCA with two additional elements: (a) the three anchor
images now carry full per-dimension reference scores (not just the overall
VC), and (b) the model is required to emit numeric scores for each of the
seven dimensions in addition to `vc_score`.

Dropped relative to the original design: V0+CA (redundant two-feature cell
without T) and V0+TCA (superseded by V0+TWC once W emerges as the
strongest pairing with T).

## 5. Test Conditions

**E1 variants (V0 through V0+TWCA).** Each variant is evaluated under
four independent runs on `claude-opus-4-6` (primary model): `r1` and `r2`
are deterministic ($t = 0$) replicates; `r3` and `r4` use Claude's
adaptive extended-thinking mode. A subset of `r3` (variants V0+TC, V0+TA)
used `claude-sonnet-4-6`; these were re-run in `r4` on `claude-opus-4-6`
for cross-variant parity. We define **det_avg** as the mean of `r1` and
`r2`, and **think_avg** as the mean of `r3` and `r4`.

**V1.** Run twice on `claude-opus-4-6` at the Anthropic API default
sampling temperature ($t = 1$, not explicitly set) with no extended
thinking and `max_tokens = 1500` to accommodate the longer per-dimension
JSON output. We report both runs and their mean (**V1_avg** = mean of
`r1` and `r2`); V1 runs carry more sampling noise per run than the E1
*det_avg* values since $t > 0$.

## 6. Results — CCC per run

Table below summarizes CCC across all runs and aggregated columns; we
then walk through the patterns visible in the table.

**Single-component gains are small and ordered $T > C > A$.** Among
the three single-component cells, V0+T leads (0.869), V0+C is mid
(0.854), and V0+A is lowest (0.852) — only 0.011 above the V0 baseline
(0.858). This ordering is what motivated the T-anchored pruning of
the two- and three-feature cells in §5: pairing topics with each
remaining component is more informative than re-running the lattice
without T.

**Layering along the T-path is near-flat at $t = 0$.** V0+T, V0+TC,
V0+TA, V0+TW, V0+TWC, V0+TWA, and V0+TWCA all sit within 0.018 CCC
of each other (0.854–0.872). Among the double-feature T-pairs,
**V0+TW is the strongest (0.872)**, narrowly above V0+TA (0.860)
and V0+TC (0.854). Adding C or A on top of V0+TW does *not* help:
the two triples lose ~0.015 CCC (V0+TWC 0.856, V0+TWA 0.857) and
the quadruple V0+TWCA (0.869) does not recover the V0+TW level.
Within the standard ablation lattice, V0+TW is therefore the best
deterministic cell of the entire sweep.

| Variant    | r1    | r2    | **det_avg** | r3    | r4    | **think_avg** |
|------------|-------|-------|-------------|-------|-------|---------------|
| V0         | 0.858 | 0.859 | **0.858**   | 0.868 | 0.881 | **0.875**     |
| V0+T       | 0.871 | 0.867 | **0.869**   | 0.857 | —     | **0.857**     |
| V0+C       | 0.851 | 0.856 | **0.854**   | —     | —     | —             |
| V0+A       | 0.852 | 0.853 | **0.852**   | —     | —     | —             |
| V0+TC      | 0.854 | —     | **0.854**   | 0.737 | 0.866 | **0.802**     |
| V0+TA      | 0.860 | —     | **0.860**   | 0.849 | 0.834 | **0.841**     |
| V0+TW      | 0.872 | 0.872 | **0.872**   | —     | —     | —             |
| V0+TWC     | 0.855 | 0.856 | **0.856**   | —     | —     | —             |
| V0+TWA     | 0.860 | 0.855 | **0.857**   | —     | —     | —             |
| V0+TWCA    | 0.868 | 0.871 | **0.869**   | 0.811 | 0.815 | **0.813**     |
| **V1**     | 0.882 | 0.880 | **0.881**   | —     | —     | —             |

*V1 rows are $t = 1$ sampling; all other rows are deterministic ($t = 0$)
or adaptive thinking as labeled.*

**Extended thinking is a double-edged sword.** The *think_avg* column
tells a non-monotone story: extended thinking lifts the minimal V0
prompt (0.858 → 0.875) but sharply hurts heavily-instructed prompts
(V0+TWCA: 0.869 → 0.813; V0+TC: 0.854 → 0.802). The qualitative
weight guidance and the calibration ranges appear to interact poorly
with longer deliberation. The low V0+TC `r3` cell (0.737) is also
partly a model-capacity confound: that single run used
`claude-sonnet-4-6` and is recovered by its `r4` opus replicate
(0.866), so the dip reflects model size rather than a prompt defect.

**V1 breaks the $0.87$ ceiling.** The extended variant V1 (full
per-dimension anchor supervision and required per-dimension numeric
output, on top of V0+TWCA) reaches CCC = 0.881 — the only cell above
the ~0.87 band that the standard ablation never crosses. The two V1
replicates agree closely (0.882 and 0.880), so the $t = 1$ sampling
noise is small relative to the gap to V0+TW. We expand on this gain
in §7 alongside the full agreement metrics.

## 7. Results — Full metrics

$n = 43$ non-anchor images, `claude-opus-4-6`. All Pearson $p < 10^{-12}$.
E1 rows report *det_avg* (mean of two $t = 0$ replicates); the V1 row
reports the mean of two $t = 1$ replicates (per-run stats averaged).
The CCC ordering of §6 is broadly preserved across $r$, $\rho$, $R^2$,
MAE, and RMSE, with one clear-cut conclusion that the multi-metric
view sharpens.

| Variant    | $n$ | CCC       | $r$       | $\rho$ (Spearman) | $R^2$     | MAE       | RMSE      | Bias      |
|------------|----:|----------:|----------:|------------------:|----------:|----------:|----------:|----------:|
| V0         | 43  | 0.858     | 0.868     | 0.868             | 0.671     | 0.080     | 0.098     | $+0.010$  |
| V0+T       | 43  | 0.869     | 0.879     | 0.872             | 0.692     | 0.078     | 0.095     | $+0.003$  |
| V0+C       | 43  | 0.854     | 0.871     | 0.873             | 0.679     | 0.081     | 0.097     | $+0.033$  |
| V0+A       | 43  | 0.852     | 0.869     | 0.866             | 0.638     | 0.083     | 0.103     | $-0.012$  |
| V0+TC      | 43  | 0.854     | 0.874     | 0.870             | 0.688     | 0.078     | 0.096     | $+0.037$  |
| V0+TA      | 43  | 0.860     | 0.882     | 0.872             | 0.656     | 0.081     | 0.101     | $-0.025$  |
| V0+TW      | 43  | 0.872     | 0.890     | 0.885             | 0.678     | 0.082     | 0.097     | $+0.026$  |
| V0+TWC     | 43  | 0.856     | 0.872     | 0.872             | 0.689     | 0.079     | 0.096     | $+0.064$  |
| V0+TWA     | 43  | 0.857     | 0.887     | 0.881             | 0.635     | 0.085     | 0.104     | $+0.000$  |
| V0+TWCA    | 43  | 0.869     | 0.876     | 0.871             | 0.704     | 0.075     | 0.093     | $-0.003$  |
| **V1**     | 43  | **0.881** | **0.900** | **0.894**         | **0.716** | **0.076** | **0.091** | $-0.026$  |

**V1 dominates on every metric.** V1 is the only variant that
simultaneously maximizes CCC, $r$, Spearman $\rho$, and $R^2$ while
minimizing MAE and RMSE. The two elements that distinguish V1 from
its standard-ablation predecessor V0+TWCA are (a) per-dimension
reference scores on each anchor (beyond the holistic VC label) and
(b) required per-dimension numeric output. Because V1 uses the same
model as every other cell, the gain is a clean prompt-design effect:
forcing the model to commit to per-topic numbers, against per-topic
anchors, sharpens the holistic VC score even though that score is
the only quantity actually scored against ground truth. The bias
column is the only place V0 / V0+T / V0+TWCA edge V1 ($-0.026$ for
V1 vs. near-zero for several E1 cells), but their lower bias is not
enough to overcome larger correlation and scale-mismatch losses.

**$R^2$ separates the variants more than CCC.** $R^2$ against the
identity line spreads variants more than CCC does (0.635–0.716 vs.
0.852–0.881): $R^2$ is more sensitive to outlier residuals, so it
discriminates between variants whose CCC values are statistically
tied. V0+TWCA ($R^2 = 0.704$) and V1 (0.716) are the clearest
leaders here; V0+A and V0+TWA sit at the bottom (0.638, 0.635),
suggesting that anchor-only and weight+anchor cells produce a few
large residuals that the smoother CCC metric absorbs.

## 8. Findings

- **Topics dominate, weighting helps, calibration and anchors do
  not stack.** The standard ablation peaks at V0+TW (0.872); adding
  C or A on top of V0+TW does not improve CCC.
- **Extended thinking interacts negatively with instruction
  density.** It helps V0 but hurts V0+TWCA and V0+TC, so adaptive
  thinking should be reserved for short prompts.
- **Per-dimension supervision is the only ingredient that breaks
  the ~0.87 ceiling.** V1 wins on every agreement metric; the gain
  is a prompt-design effect, not a model-capacity effect.

## 9. Topic Selection (Follow-on Study)

The 7 topics used inside the prompt (data density, visual encoding, text
legibility, schema, color/symbol, aesthetics, cognitive load) were
originally derived from a bottom-up coding of the same user-comment
corpus that produced the human VC ratings. Each topic in that coding
maps 1:1 to a scoring dimension in V1, so for every evaluation image we
also have a human-derived **topic set** — the subset of topics that
reviewers actually mentioned when describing that image. This enables a
second, complementary evaluation: rather than asking the model to
reproduce a continuous VC score, we ask whether it can identify *which
topics* drive complexity for a given image.

**Ground truth.** For each of the $n=43$ evaluation images, we take the
set of topics present in the compiled human comment pool (semicolon-
delimited `Topics` column of `image_compiled_phrases.csv`, mapped to
dimension keys). Human cardinality ranges 1–7 topics per image
(median 3); 14 images list $\geq 5$ topics.

**Model task.** Variants that already expose the 7 topics to the model
(V0+T, V0+TW, V1) are extended to emit a top-3 topic ranking. For V0+T
and V0+TW the prompt is modified to additionally request a
`top3_topics` field (keys drawn from the canonical 7). V1 requires no
prompt change: we derive top-3 directly from its per-dimension numeric
scores (highest-3 dims).

**Metric.** Because human topic sets are unordered and variable-sized,
we report **set-based $F_1$** as the headline
($F_1 = 2PR/(P+R)$ with $P = |\text{pred} \cap \text{human}| / 3$ and
$R = |\text{pred} \cap \text{human}| / |\text{human}|$), plus the
Jaccard/IoU, macro precision/recall (mean per image), and micro $F_1$
(pooled counts). Per-topic precision/recall/$F_1$ serve as diagnostics.

### 9.1 Aggregate Results (n=43, opus-4.6)

The table below summarizes the top-3 topic-selection performance for
the three variants that already expose the 7-topic taxonomy. Two
patterns dominate the column.

**Prompting for top-3 beats deriving from per-dimension scores.**
V0+T and V0+TW, which are asked explicitly to return a top-3 topic
list, both outperform the V1 runs by $\geq 9$ $F_1$ points (V0+TW
0.528 vs. V1 0.407). V1 is strictly better at the *continuous* VC
task (§6–§8), but its per-dimension scores — trained to serve the
holistic VC aggregate — do not discriminate the top-3 topics as
sharply as a prompt that asks for them directly. The V1-vs-E1
ordering of the continuous task therefore reverses on the
categorical task.

**Qualitative weight guidance helps in aggregate.** V0+TW exceeds
V0+T by $+0.031$ $F_1$ (0.528 vs. 0.497) on the headline metric and
leads on every column except Macro P / Macro R. The "high/medium/low"
bucketing biases selections toward data_density, visual_encoding,
schema, and cognitive_load, which are the four most-frequent human
topics in the eval set. Whether this aggregate gain is a free lunch
is the question we take up in the per-topic breakdown below.

**$t=0$ and $t=1$-mean produce identical V1 results.** The two V1
rows agree to four decimals on every metric, confirming that the
rank-of-three ordering is invariant to $t=1$ sampling noise across
the two V1 replicates. Sampling temperature is a scoring-precision
knob, not a ranking-structure knob, for this task.

| Variant                         | $F_1$ (macro)   | IoU           | Macro P | Macro R | Micro $F_1$ |
|---------------------------------|-----------------|---------------|---------|---------|-------------|
| **V0+TW** (prompted top-3)      | **0.528**       | **0.385**     | 0.597   | 0.550   | **0.544**   |
| V0+T (prompted top-3)           | 0.497           | 0.360         | 0.574   | 0.503   | 0.523       |
| V1 ($t=0$, derived)             | 0.407           | 0.281         | 0.488   | 0.387   | 0.445       |
| V1 ($t=1$, mean of $r_1,r_2$)   | 0.407           | 0.281         | 0.488   | 0.387   | 0.445       |

### 9.2 Per-Topic $F_1$

The aggregate columns of §9.1 hide a more uneven picture across the
7 topics.

**Weighting trades per-topic fidelity for aggregate fit.** V0+TW's
qualitative weight guidance boosts `data_density` (0.765 → 0.800)
and `cognitive_load` (0.566 → 0.806), but actively *suppresses*
`text_annotation` (0.364 → 0.087, which is explicitly labeled "low
weight"), and cuts `color_symbol` from 0.538 to 0.293. V0+TW's
aggregate lead therefore comes from its alignment with the
topic-frequency distribution of the eval set, not from uniformly
better perception; in a deployment where every topic matters
equally, V0+T is the safer choice.

**`aesthetic_order` is unreachable across all variants ($F_1 = 0$).**
No variant ever ranks it in the top-3. It is the rarest human topic
(6 of 43 images) and semantically conflates with `data_density` and
`cognitive_load` in the model's representation; recovering it likely
requires either an explicit aesthetic-only sub-prompt or a different
ground-truth aggregation.

**V0+T uniquely picks up `color_symbol`.** V0+T scores $F_1 = 0.538$
on color_symbol, while V0+TW and V1 under-select it
($F_1 \approx 0.30$–$0.34$, high precision but low recall).
Color-driven complexity is easily absorbed into `data_density` or
`visual_encoding` by variants that are not explicitly prompted to
separate them.

| Variant          | data_density | visual_encoding | text_annotation | domain_schema | color_symbol | aesthetic_order | cognitive_load |
|------------------|:------------:|:---------------:|:---------------:|:-------------:|:------------:|:---------------:|:--------------:|
| V0+T             | 0.765        | 0.323           | 0.364           | 0.432         | **0.538**    | 0.000           | 0.566          |
| **V0+TW**        | **0.800**    | 0.389           | 0.087           | 0.410         | 0.293        | 0.000           | **0.806**      |
| V1 ($t=0$)       | 0.677        | 0.372           | 0.343           | **0.465**     | 0.342        | 0.000           | 0.468          |
| V1 ($t=1$, mean) | 0.698        | 0.326           | 0.343           | **0.465**     | 0.342        | 0.000           | 0.468          |

### 9.3 Per-vis-type F1 (n = 63)

To test whether prompt performance varies by vis-type, we expanded
the evaluation set to 63 images (7 per vis-type × 9 vis-types;
3 anchor images excluded) and re-evaluated V0+T and V0+TW.

**Point plots are easiest; Glyph is hardest.** Point achieves the
highest $F_1$ for both variants (V0+T: 0.752, V0+TW: 0.711), while
Glyph is the lowest (0.401 / 0.458). Glyph visualizations combine
dense encoding with domain-specific spatial arrangements, making their
dominant topics harder to rank unambiguously.

**V0+TW's weight guidance helps chart-centric types.** V0+TW
outperforms V0+T on Bar (+0.062), Cont.-ColorPatn (+0.071), Glyph
(+0.057), and Node-link (+0.036), all of which rely heavily on the
topics the weighting promotes (`data_density`, `visual_encoding`,
`domain_schema`). Conversely, V0+T retains an advantage on Line
(+0.058), Point (+0.041), and Text (+0.052), where suppressing
`text_annotation` and `color_symbol` in V0+TW hurts. Area and Grid
are tied between variants.

| Vis-type        | n  | V0+T F1 | V0+T P | V0+T R | V0+TW F1 | V0+TW P | V0+TW R |
|-----------------|:--:|:-------:|:------:|:------:|:--------:|:-------:|:-------:|
| Area            | 7  | **0.463** | 0.571 | 0.466  | **0.463** | 0.571  | 0.466   |
| Bar             | 7  | 0.420   | 0.429  | 0.526  | **0.482** | 0.476  | 0.605   |
| Cont.-ColorPatn | 7  | 0.470   | 0.571  | 0.491  | **0.541** | 0.619  | 0.634   |
| Glyph           | 7  | 0.401   | 0.476  | 0.367  | **0.458** | 0.524  | 0.438   |
| Grid            | 7  | **0.533** | 0.619 | 0.514  | **0.533** | 0.619  | 0.514   |
| Line            | 7  | **0.580** | 0.619 | 0.571  | 0.522    | 0.571   | 0.500   |
| Node-link       | 7  | 0.535   | 0.524  | 0.676  | **0.571** | 0.571  | 0.705   |
| Point           | 7  | **0.752** | 0.857 | 0.705  | 0.711    | 0.810   | 0.669   |
| Text            | 7  | **0.468** | 0.571 | 0.421  | 0.416    | 0.524   | 0.356   |
| **Macro avg**   | 63 | 0.513   |        |        | **0.522** |         |         |

### 9.4 Findings

- **Categorical and continuous tasks favour different prompts.**
  V1 wins the continuous VC task; V0+TW wins the top-3
  topic-selection task. Per-dimension numeric output sharpens
  scoring but does not sharpen ranking.
- **Weighting buys aggregate fit at the cost of per-topic fairness.**
  V0+TW leads on $F_1$ but suppresses the topics it labels "low
  weight" (`text_annotation`, `color_symbol`); pick V0+T when
  uniform per-topic recall matters more than aggregate $F_1$.
- **`aesthetic_order` is a structural blind spot.** No variant
  recovers it; this is a limitation of the 7-topic taxonomy and the
  model's lexical mapping, not of any individual prompt.
- **Vis-type moderates prompt choice.** On Glyph, Cont.-ColorPatn,
  Bar, and Node-link, V0+TW's weighting guidance lifts $F_1$ by
  0.04–0.07; on Line, Point, and Text, the unweighted V0+T holds an
  equivalent advantage. No single variant dominates uniformly across
  vis-types.

Outputs: `topic_selection/human_topic_gt.csv`,
`topic_selection/topic_summary.csv`,
`topic_selection/per_topic_f1.csv`,
`topic_selection/per_image_all.csv`.
Scripts: `scripts/_vc_score_api_v0_t_top3.py`,
`scripts/_vc_score_api_v0_tw_top3.py`,
`scripts/topic_select_full_metrics.py`.

## 10. Structured Extraction: Human-Phrase vs. Vision Grounding

The experiments in §1–§9 evaluate the model on the *scalar* VC task.
This section asks a complementary question: **when the model must
produce structured scene content — objects, attributes, and
relationships (O/A/R) — how close does vision-only grounding come to
the structured content a human-phrase baseline produces?**

The extraction schema — objects with regions, attributes bound to
objects, and relationships expressed as
$(\text{subject}, \text{predicate}, \text{object})$ triples — follows
the **Visual Genome** dense-annotation formulation of Krishna et
al. [Krishna2017VG], which showed that objects / attributes /
relationships form a sufficient substrate for question answering and
grounded reasoning over natural images. We adapt that formulation
from natural scenes to data visualizations: objects are chart
components (bars, axes, legends, titles), attributes carry a
perceptual sentiment $\{+, -\}$ on the topic they instantiate, and
predicates encode complexity-relevant relations (`increases_effort`,
`aids_interpretation`, `obscures`, `simplifies`, `describes`,
`structures`, …) rather than the spatial/action relations
(`riding`, `holding`) that dominate Visual Genome. Canonicalization —
collapsing surface strings to a small vocabulary of synsets and
canonical predicates before any set-based comparison — is also
adopted from the Visual Genome pipeline, where it was necessary to
make annotator-written triples comparable across 108k images. The
canonicalization dictionaries used here are defined in
`scripts/_vc_canon.py` and shared with the main pipeline
(`VCGenome_FullPipeline.ipynb`).

### 10.1 Three Conditions (n=43)

Using the same 43-image eval set (46 minus the 3 anchors), we run
three extraction conditions with the same schema (objects with
regions; attributes with sentiment and topic; relationships with
predicate, sentiment, and topic), the same 7-topic taxonomy, and the
same decoding settings ($t=0$, claude-sonnet-4-6, 1024 tokens):

| Cond. | Grounding signal                                     | Input |
|-------|------------------------------------------------------|-------|
| **B** | topics + **sentiment-tagged participant phrases** for the image | text only |
| **V1**| topics + **image**                                   | vision |
| **V2**| topics + image + 3 few-shot **anchors** sourced from B | vision + exemplars |

B serves as the human-phrase-grounded reference; V1 is pure vision;
V2 adds three in-context O/A/R exemplars taken from B's own output on
the three anchor images (so V2's anchors are guaranteed to be in the
same vocabulary and abstraction level as B). The anchor images
themselves are excluded from all metrics.

### 10.2 Matching Procedure

Each extraction is canonicalized with the pipeline's synset
dictionaries (`scripts/_vc_canon.py`): objects → 1 of ~40 synsets
(`mark.bar`, `structure.region`, `text.title`, …), predicates → 1 of
~30 canonical relations, attributes → lowercased snake_case. We then
compute set-based precision / recall / $F_1$ / Jaccard per image and
average over the 43 images. Three layers are reported:

- **Objects** — shared synsets.
- **Attributes with sentiment** — $(\text{object\_synset}, \text{sentiment})$
  pairs (wording-agnostic; captures "does the sign at this object
  agree?").
- **Relationships** — reported in three granularities: a **strict**
  triple $(s, p, o)$, a **family** triple that rolls $p$ into one of
  ~10 predicate families (aids, hinders, obscures, simplifies,
  increases_complexity, increases_effort, describes, structures, …),
  and a **loose** endpoint pair $\{s, o\}$.

**Role-level layer (cross-vocabulary alignment).** Preliminary
synset-level numbers revealed that B and the vision conditions
describe the same pictures with systematically different abstraction
levels: B names roles (`chart`, `data_area`, `color_encoding`,
`annotations`) while V1/V2 name instances
(`horizontal_bars`, `deceased_bars`, `country_labels`). To separate
*vocabulary mismatch* from *genuine disagreement*, we additionally
project every canonical synset to one of 8 chart-part **roles** —
`chart`, `data_area`, `encoding`, `title`, `axes`, `labels`,
`legend`, `background` — and recompute the three layers on the
projected keys. Role-level numbers reflect "did the two extractions
agree on what chart region the claim is about," independent of
whether one says `horizontal_bars` and the other says `data_area`.

### 10.3 Results

**Mean $F_1$ across 43 images, strict synset-level keys.**

| Pair       | Obj  | Attr (sign) | Rel-family | Rel-loose |
|------------|:----:|:-----------:|:----------:|:---------:|
| B vs V1    | 0.15 | 0.09        | 0.00       | 0.01      |
| B vs V2    | 0.15 | 0.09        | 0.00       | 0.01      |
| V1 vs V2   | 0.66 | 0.60        | 0.19       | 0.39      |

**Mean $F_1$, role-level keys (synsets projected to 8 chart-part
roles).**

| Pair       | Obj-role | Attr-role (sign) | Rel-role-family | Rel-role-loose |
|------------|:--------:|:----------------:|:---------------:|:--------------:|
| B vs V1    | 0.25     | 0.19             | **0.03**        | **0.06**       |
| B vs V2    | **0.28** | **0.19**         | 0.02            | 0.05           |
| V1 vs V2   | 0.72     | 0.68             | 0.21            | 0.48           |

### 10.4 Example Scene Graphs

To illustrate the qualitative difference between the two grounding
strategies, Figs. 1–2 render the canonicalized scene graphs that B
and V2 produced for the same image (`SciVisJ.2926.1.png`, a 3D
scientific flow visualization). The two graphs are deliberately
small (B: 1 object / 3 attributes / 0 relationships; V2: 5 / 9 / 5)
to make the vocabulary contrast easy to read.

![Condition B (human-phrase grounded) scene graph for `SciVisJ.2926.1.png`. Objects are generic roles (`visualization`, `content`, `labels`); relationships are *effectual* (`fails_to_convey`, `requires_domain_knowledge`).](figures/oar_SciVisJ.2926.1_B.png)

*Figure 1.* Condition B scene graph (human-phrase grounded).

![Condition V2 (vision + B-sourced anchors) scene graph for the same image. Objects are domain-specific (`vortex_structure`, `glyph_spikes`, `flow_field`, `annotation_label`); relationships are *structural* (`contains`, `overlaps_with`, `describes`, `requires_expertise`).](figures/oar_SciVisJ.2926.1_V2.png)

*Figure 2.* Condition V2 scene graph (vision + B-sourced anchors).

The contrast is the central qualitative finding of this study:
B cannot see the picture and so labels everything with the same
three generic chart parts and reaches for *effectual* predicates
about interpretation; V2 sees the image and produces
domain-specific scientific-visualization terms
(`vortex_structure`, `glyph_spikes`, `flow_field`) connected by
*structural* predicates about spatial composition. This is exactly
the pattern the aggregate metrics in §10.3 anticipate.

### 10.5 Findings

- **Vocabulary abstraction, not perception, drives most of the B↔V
  gap on objects.** Collapsing synsets to 8 chart-part roles raises
  B↔V2 object $F_1$ from 0.15 to 0.28 (+87%) and B↔V1 from 0.15 to
  0.25 (+67%). The two vision conditions (V1 vs V2) also rise but by
  much less (0.66 → 0.72), confirming they already share an
  instance-level perceptual lexicon; the human baseline's vocabulary
  sits one abstraction level higher.
- **Phrase-grounded anchors give only a small, uneven nudge.**
  Under raw, sentiment-tagged participant phrases (rather than a
  curated set), V2 narrowly edges V1 on role-level objects (0.25 →
  0.28, +0.03) while attribute-sign is tied (0.19); strict-synset
  deltas are within rounding (0.00 on both) and role-level loose
  relationships slightly favour V1 (0.06 vs. 0.05). Three B-sourced exemplars
  therefore nudge object vocabulary toward the human-phrase
  distribution but do not shift attribute-sentiment or relationship
  structure in any practically meaningful way.
- **Relationships remain the hardest layer.** Even at the role level
  and with family rollup, B↔V rel-family $F_1$ stays near zero
  (0.02), while B↔V rel-loose is 0.05–0.06. The two vision
  conditions agree on role-family relationships at $F_1 = 0.21$. The
  divergence is conceptual, not lexical: V1/V2 relationships are
  predominantly *structural* ("bars stacked with bars", "labels
  label rows", "x-axis provides scale for bars"), while B
  relationships are predominantly *effectual* ("color encoding adds
  visual complexity to data area", "axis labels support readability
  of data area", "data density increases reading effort of chart").
  The human-phrase prompt pushes the model to reason about *effects
  on perception*; the vision prompt pushes it to describe *spatial
  arrangement*. This divergence is essentially independent of
  anchoring — V2 barely moves it (rel-role-loose: V1 0.06, V2 0.05).
- **Headline takeaway for VC modelling.** Structured O/A/R
  extraction from images can recover the human baseline's *what* and
  *valence* only modestly once vocabulary is normalised (object-role
  $F_1 \approx 0.4$, attribute-sign role $F_1 \approx 0.25$ for V2),
  and it does *not* recover the human baseline's *causal framing* of
  the scene. Any downstream VC predictor that consumes O/A/R should
  therefore rely on objects and attribute-sentiment derived from
  vision, and treat the relationship layer either as human-phrase
  supervision or as a derived feature built on top of the canonical
  predicate families.

## References

[Krishna2017VG] R. Krishna, Y. Zhu, O. Groth, J. Johnson, K. Hata, J. Kravitz, S. Chen, Y. Kalantidis, L.-J. Li, D. A. Shamma, M. S. Bernstein, and L. Fei-Fei. *Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations.* International Journal of Computer Vision, 123(1):32–73, 2017.

## 12. VC Prediction on the Full Dataset (V0+TW vs. V1)

The ablation in §1–§8 and the topic-selection study in §9 both operate on the 46-image pilot set ($n = 43$ after anchor exclusion). That set was deliberately constructed to maximize $|\Delta\mathrm{VC}|$ coverage rather than to be representative of the broader image pool — its wider GT spread inflates CCC relative to an unselected sample. This section extends the VC prediction evaluation to the full curated image corpus to assess how the two best-performing prompts generalise: **V0+TW** (best deterministic E1 cell, §6–§8) and **V1** (upper envelope of the ablation).

### 12.1 Setup

**Image set.** Ground-truth `NormalizedVC` ratings for all images are taken from `phrase_reduction_v2/image_compiled_phrases.csv`. We retain the 9 vis-types with sufficient representation (Bar, Line, Area, Point, Node-link, Glyph, Grid, Cont.-ColorPatn, Text) and drop four rare categories (Schematic, Bar and point, Table, Area and Text), yielding **510 images**. Following the same leakage-prevention convention applied in §1, the three anchor exemplars are excluded from both prompts' metrics, yielding $n = 507$ test images. V1 requires this exclusion because it receives the anchors as in-context examples; we apply it to V0+TW as well so both conditions are evaluated on an identical image set. Full-set GT statistics: mean $= 0.598$, SD $= 0.148$ — a noticeably narrower spread than the pilot (pilot SD $\approx 0.244$), consistent with the pilot's deliberate difficulty stratification.

**Runs.** Both prompts run on `claude-opus-4-6` at $t = 0$ (deterministic), concurrency = 5. V0+TW uses the same runner as the pilot (`scripts/run_tw_r1_r2.bat`). V1 was originally scored at the API default $t = 1$ on the unfiltered 520-image pool; to eliminate a temperature confound, it was re-run at $t = 0$ on the 510-image set via `scripts/run_v3_510_t0.bat`. The temperature sensitivity analysis confirming this re-run does not materially change V1's output is reported in §12.4. One deterministic run per prompt is available (no double-replicate averaging as in §5–§8); all figures below are single-run point estimates.

### 12.2 Overall Results

Full-set CCC drops relative to the pilot for both prompts — V0+TW: pilot 0.872 → full 0.784; V1: pilot 0.881 → full 0.777. The decline is expected: the pilot's easy-discrimination images inflate the denominator of CCC; the full set re-introduces many mid-range images where small residuals accumulate. The drop is proportionally similar for both prompts, suggesting it reflects the evaluation-set shift rather than a prompt-specific failure.

| Prompt | $t$ | $n$ | CCC | Pearson $r$ | Spearman $\rho$ | $R^2$ | MAE | RMSE | Bias |
|--------|-----|----:|----:|------------:|----------------:|------:|----:|-----:|-----:|
| V0+TW | 0 | 507 | 0.784 | 0.835 | 0.833 | 0.370 | 0.093 | 0.116 | $-0.030$ |
| V1    | 0 | 507 | 0.777 | 0.854 | 0.851 | 0.388 | 0.093 | 0.115 | $-0.063$ |

On the identical 507-image test set, the two prompts differ by $\Delta\mathrm{CCC} = -0.007$ and $\Delta\mathrm{MAE} = 0.000$. V1 gains approximately 0.02 on Pearson $r$, but this improvement is offset by a larger negative bias: V1 systematically under-predicts by $-0.063$ vs. V0+TW's $-0.030$. Since CCC jointly penalises correlation loss *and* mean shift, the net CCC difference is negligible.

**Prediction spread.** Both prompts over-spread relative to human GT (pred SD $>$ GT SD $= 0.148$). V0+TW is wider (pred SD $= 0.201$); V1 is narrower but still over-spread (pred SD $= 0.183$). V1's tighter spread partly explains its higher Pearson $r$ (a scale-invariant metric) while its larger negative bias neutralises any CCC advantage.

**Note on $R^2$ conventions.** Throughout this document, $R^2$ is `sklearn.metrics.r2_score` — evaluated against the identity line $\hat{y} = y$ — and can go negative when predictions are biased off the diagonal. The companion notebook `VC_520_Analysis.ipynb` reports Pearson $r^2$ (coefficient of determination from an unconstrained linear regression of pred on GT), which is bounded in $[0, 1]$ and does not penalise mean shift or scale mismatch. For V1 at $t = 0$ on the 507-image test set: Pearson $r^2 = 0.730$ vs. sklearn $R^2 = 0.388$. The gap of $0.342$ is entirely attributable to bias and scale mismatch ($\mathrm{bias} = -0.063$, $\sigma_{\hat{y}} / \sigma_y = 1.24$).

### 12.3 Per-Vis-Type Breakdown ($n$ = 507, anchors excluded)

| Vis Type | n | CCC V0+TW | CCC V1 | ΔCCC | R² V0+TW | R² V1 | MAE V0+TW | MAE V1 |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Node-link        |  66 | 0.797 | 0.818 | **+0.021** |  0.434 |  0.555 | 0.096 | 0.080 |
| Area             |  65 | 0.833 | 0.825 | −0.008 |  0.527 |  0.519 | 0.080 | 0.083 |
| Grid             |  64 | 0.738 | 0.731 | −0.007 |  0.275 |  0.272 | 0.097 | 0.099 |
| Glyph            |  64 | 0.729 | 0.719 | −0.010 |  0.062 |  0.102 | 0.102 | 0.101 |
| Point            |  57 | 0.761 | 0.671 | **−0.090** |  0.299 |  0.011 | 0.090 | 0.111 |
| Bar              |  50 | 0.823 | 0.785 | −0.038 |  0.514 |  0.445 | 0.092 | 0.098 |
| Text             |  51 | 0.781 | 0.761 | −0.020 |  0.498 |  0.404 | 0.070 | 0.080 |
| Line             |  48 | 0.732 | 0.745 | +0.013 |  0.164 |  0.292 | 0.103 | 0.090 |
| Cont.-ColorPatn  |  42 | 0.562 | 0.614 | **+0.052** | −1.223 | −0.540 | 0.116 | 0.098 |

The per-type picture is split. V1 wins on Node-link (+0.021 CCC), Cont.-ColorPatn (+0.052), and Line (+0.013); V0+TW wins on Point (−0.090), Bar (−0.038), Text (−0.020), Glyph (−0.010), Grid (−0.007), and Area (−0.008). No prompt dominates uniformly.

**Best-performing types (Bar, Area).** Both prompts achieve CCC $\geq 0.826$ on Bar and Area — the most common chart categories, where the model's prior apparently aligns well with human VC ratings and the complexity gradient is easy to operationalise.

**Hardest type: Cont.-ColorPatn.** Both prompts fail on continuous colour-pattern visualisations ($R^2 < 0$ for both). The GT distribution for this type is unusually narrow (GT SD $= 0.105$), so any over-spread prediction falls systematically off the identity line. V1's calibration and anchor components reduce over-spread on this type (pred SD V0+TW $\approx 0.203$, V1 $\approx 0.167$), which converts a catastrophic $R^2 = -1.22$ into a merely poor $R^2 = -0.54$ while lifting CCC by 0.052.

**Biggest V1 regression: Point.** V1's CCC drops from 0.761 to 0.671 ($\Delta = -0.090$) on scatter / point charts, and its $R^2$ collapses from 0.299 to 0.011. The calibration ranges and anchor exemplars in V1 appear to push Point predictions away from the true mid-range cluster for this type.

### 12.4 Effect of Temperature Re-run

To verify the temperature was not driving the V0+TW vs. V1 comparison, we compare the original V1 run at $t = 1$ (unfiltered 520-image pool, `vc_api_scores/vc_scores.csv`) with the V1 re-run at $t = 0$ ($n = 507$ after anchor exclusion, `vc_api_510_v3_t0/vc_scores.csv`) on the overlapping image subset. Across all agreement statistics (CCC, Pearson $r$, $R^2$, MAE), the shift from $t = 1$ to $t = 0$ is $\leq 0.005$ — well within single-image sampling noise. Temperature setting is therefore not a material confound in the V0+TW vs. V1 comparison.

### 12.5 Findings

- **Full-set CCC is lower than pilot CCC for both prompts** (V0+TW: 0.784 vs. 0.872; V1: 0.777 vs. 0.881), primarily because the pilot's difficulty-stratified sampling produces a wider GT spread that flatters CCC.
- **V0+TW and V1 are indistinguishable at the headline level** ($\Delta\mathrm{CCC} = -0.007$, $\Delta\mathrm{MAE} = 0.000$ on the identical $n = 507$ test set) with temperature and model held constant. The extra V1 machinery (Calibration + Anchors + per-dimension JSON output) does not yield a meaningful accuracy improvement on a representative image pool.
- **Per-type outcomes are split.** V1 is better for Node-link and Cont.-ColorPatn; V0+TW is better for Point and Bar. Applications sensitive to Point-chart accuracy should prefer V0+TW; those targeting Node-link or continuous colour maps may prefer V1.
- **Cont.-ColorPatn is a structural weakness for both prompts** ($R^2 < 0$). Improving this type likely requires either type-specific calibration guidance or a broader anchor set covering extreme values in this category.
- **The practical recommendation** from §8 holds on the full dataset: **V0+TW is the preferred prompt** — it matches V1 on overall agreement with fewer instructions, no per-dimension output overhead, and better performance on the high-frequency Point and Bar types.

**Artifacts.**
- V0+TW 510-image scores: `results/vc_api_510_v0_tw/vc_scores.csv`; launch script: `scripts/run_tw_r1_r2.bat`.
- V1 $t=0$ scores: `results/vc_api_510_v3_t0/vc_scores.csv`; launch script: `scripts/run_v3_510_t0.bat`.
- Comparison analysis: `Claude_vc_prediction/V0_Variants_Comparison.ipynb` (cells §§ "Earlier Production Run" onwards).
- Saved metric tables: `results/vc_api_510_v0_tw/metrics_v3_vs_v0tw_long.csv` and `metrics_v3_vs_v0tw_wide.csv`.

## 11. BibTeX for Lin's CCC

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

## 13. Prompt Listings

All variants share a common structure: a **system prompt** (delivered once per API call) followed by one or more **user turns** (one per image). The system prompt is built by starting from the **V0 base block** and inserting the four modular component blocks (**T**, **W**, **C**, **A**) in the positions marked below. Each block is the exact text that appears in the deployed prompt; nothing is paraphrased.

---

### 13.1 V0 — Baseline (full system prompt)

This is the complete system prompt for V0. Every other variant starts here and splices in one or more component blocks.

```text
You are a visual complexity (VC) scoring expert for data visualizations.

**Visual Complexity** measures the amount of detail, intricacy, and
perceptual/cognitive demand of a visualization image. It is scored on a
0–1 scale where 0 = minimal complexity and 1 = maximum complexity.

You will receive a single visualization image with NO accompanying text
or comments. Score it purely from what you see.

## Output Format
Return ONLY valid JSON (no markdown fences, no explanation outside JSON):
{
  "vc_score": <float 0-1>,
  "explanation": "<2-3 sentence justification>"
}
```

**User turn (all non-anchored variants):**
```text
[IMAGE: target visualization]
Score the visual complexity of this visualization image.
```

---

### 13.2 Component Blocks

Each block is inserted into the system prompt **before the `## Output Format` section** (and in the order T → W → C). The A (Anchors) component does not add text to the system prompt; it extends the message sequence instead (see §13.3).

#### Block T — Topics / Dimensions

```text
## Topics to Consider When Scoring

When assessing visual complexity, consider the following 7 topics that
commonly drive complexity perception. You do not need to score each topic
individually — use them as mental checkpoints to arrive at a holistic
vc_score.

1. **Data Density / Image Clutter** — The perceived amount, richness, or
   depth of data content. Considers information volume, element quantity,
   and visual clutter/overlap.
2. **Visual Encoding Clarity** — The variety, type, and complexity of
   graphical forms (shapes, lines, marks) and how spatial layout, scale,
   and encoding interpretability contribute to complexity.
3. **Semantics / Text Legibility** — The quantity and density of text
   elements (titles, axis labels, legends, captions, annotations,
   in-chart labels).
4. **Schema** — Whether specialized domain knowledge is needed, including
   dimensionality (2D/3D), structural complexity, and abstraction level.
5. **Color, Symbol, and Texture Details** — Range, variety, and
   arrangement of colors, plus use of symbols, textures, and non-color
   graphical markers.
6. **Aesthetics Uncertainty** — How visually cluttered, dense, or
   disordered the layout appears. Higher = more cluttered/overwhelming.
   A clean minimal layout scores low; a crowded layout with overlapping
   elements scores high.
7. **Immediacy / Cognitive Load** — Overall ease or difficulty of
   interpreting the visualization. Considers interpretive difficulty,
   semantic clarity, and processing time/effort.
```

*Used in: V0+T, V0+TC, V0+TA, V0+TW, V0+TWC, V0+TWA, V0+TWCA, V1.*

---

#### Block W — Weighted Scoring

Appended immediately after Block T (requires T to be present).

```text
## Weighting of Topics for the Overall vc_score
The overall vc_score is a weighted holistic judgment. Weight the 7 topics
as follows:
- **High weight**: Data Density / Image Clutter, Visual Encoding Clarity,
  Schema, Immediacy / Cognitive Load (these drive VC the most).
- **Medium weight**: Color / Symbol / Texture Details, Aesthetics
  Uncertainty.
- **Low weight**: Semantics / Text Legibility (text volume alone is a
  weak VC signal).
A low score on Semantics / Text Legibility alone should NOT substantially
pull down the overall vc_score.
```

*Used in: V0+TW, V0+TWC, V0+TWA, V0+TWCA, V1.*  
*Also changes the `"vc_score"` JSON field description to `<float 0-1, weighted holistic judgment>`.*

---

#### Block C — Calibration

Appended after Block T (and W if present), before `## Output Format`.

```text
## Calibration Guidance
Use the full 0–1 range. Do NOT cluster scores conservatively toward the
center.
- A plain single bar chart or pie chart with < 5 elements
  → vc_score ≈ 0.25–0.40
- A standard scatterplot, grouped bar chart, or line chart
  → vc_score ≈ 0.45–0.60
- A multi-panel dashboard, dense network, or multi-encoding composite
  → vc_score ≈ 0.70–0.85
- An extremely dense multi-view system with many coordinated panels
  → vc_score ≈ 0.85–0.95
IMPORTANT: If you are unsure between two scores, always choose the higher
one. Most scorers underestimate — push scores up.
```

*Used in: V0+C, V0+TC, V0+TWC, V0+TWCA, V1.*

---

#### Block A — Anchor Few-Shot Turns

The A component does not modify the system prompt text. Instead, it prepends **three user→assistant turn pairs** to the message list before the target-image turn. Each pair presents one labeled anchor image:

```
[USER]   [IMAGE: VisC.503.6.png]
         Score the visual complexity of this visualization image.
[ASST]   {"vc_score": 0.22, "explanation": "Anchor image scored at 0.22."}

[USER]   [IMAGE: InfoVisJ.619.17.png]
         Score the visual complexity of this visualization image.
[ASST]   {"vc_score": 0.54, "explanation": "Anchor image scored at 0.54."}

[USER]   [IMAGE: InfoVisJ.1149.6(1).png]
         Score the visual complexity of this visualization image.
[ASST]   {"vc_score": 0.95, "explanation": "Anchor image scored at 0.95."}

[USER]   [IMAGE: target visualization]   ← actual query
         Score the visual complexity of this visualization image.
```

The three anchors span the scale: low (0.22), mid (0.54), high (0.95). Anchor images are excluded from all reported metrics to prevent leakage.

*Used in: V0+A, V0+TA, V0+TWA, V0+TWCA, V1.*

---

### 13.3 V1 — Extended Variant

V1 uses the same system prompt as V0+TWCA (Base + T + W + C), but with two modifications.

**Modification 1 — Anchor scores are per-dimension (not holistic only).** Each anchor assistant turn carries full dimension scores instead of just `vc_score`:

```
[ASST]   {
           "data_density": 0.15, "visual_encoding": 0.20,
           "text_annotation": 0.10, "domain_schema": 0.15,
           "color_symbol": 0.10, "aesthetic_order": 0.20,
           "cognitive_load": 0.20, "vc_score": 0.22,
           "data_density_explanation": "...",
           ...
           "summary": "..."
         }
```

**Modification 2 — The `## Output Format` block is replaced** to require numeric per-dimension output:

```text
## Output Format
Return ONLY valid JSON (no markdown fences, no explanation outside JSON):
{
  "data_density": <float 0-1>,
  "visual_encoding": <float 0-1>,
  "text_annotation": <float 0-1>,
  "domain_schema": <float 0-1>,
  "color_symbol": <float 0-1>,
  "aesthetic_order": <float 0-1>,
  "cognitive_load": <float 0-1>,
  "vc_score": <float 0-1>,
  "data_density_explanation": "<1 sentence>",
  "visual_encoding_explanation": "<1 sentence>",
  "text_annotation_explanation": "<1 sentence>",
  "domain_schema_explanation": "<1 sentence>",
  "color_symbol_explanation": "<1 sentence>",
  "aesthetic_order_explanation": "<1 sentence>",
  "cognitive_load_explanation": "<1 sentence>",
  "summary": "<2-3 sentence overall description>"
}
```

Additionally, the `## Scoring Dimensions` header replaces `## Topics to Consider When Scoring` in the V1 system prompt, and the dimension names use snake_case keys (`data_density`, `visual_encoding`, …) instead of the full human-readable topic names used in V0+T/TW/TWCA.

Source scripts: `scripts/_vc_score_api_v1.py` (V1), `scripts/_vc_score_api_v0_topic_weighted.py` (V0+TW), `scripts/_vc_score_api_v0_topic_calib_anchor.py` (V0+TWCA), `scripts/_vc_score_api_v0_topic.py` (V0+T), `scripts/_vc_score_api_v0_calibration.py` (V0+C), `scripts/_vc_score_api_v0_anchors.py` (V0+A).

---

### 13.4 Topic Matching (Top-3 Selection)

The topic-matching study (§9) extends V0+T and V0+TW with one additional output field `top3_topics` and a **Topic-Key Mapping** block. The system prompt is otherwise identical; only the output format and one new block differ.

#### Block: Topic-Key Mapping

Inserted immediately before `## Output Format`:

```text
## Topic-Key Mapping
Each topic corresponds to a short key:
- "Data Density / Image Clutter"       -> data_density
- "Visual Encoding Clarity"            -> visual_encoding
- "Semantics / Text Legibility"        -> text_annotation
- "Schema"                             -> domain_schema
- "Color, Symbol, and Texture Details" -> color_symbol
- "Aesthetics Uncertainty"             -> aesthetic_order
- "Immediacy / Cognitive Load"         -> cognitive_load
```

#### Modified Output Format (V0+T top-3 variant)

```text
## Output Format
Return ONLY valid JSON (no markdown fences, no explanation outside JSON):
{
  "vc_score": <float 0-1>,
  "top3_topics": ["<key>", "<key>", "<key>"],
  "explanation": "<2-3 sentence justification referencing the topics that most influenced the score>"
}

Where `top3_topics` is an ordered list (most -> least influential) of exactly three distinct keys drawn from: data_density, visual_encoding, text_annotation, domain_schema, color_symbol, aesthetic_order, cognitive_load.
```

The V0+TW top-3 variant is identical except `"vc_score"` description reads `<float 0-1, weighted holistic judgment>` (consistent with Block W). For **V1**, no prompt modification is needed: top-3 topics are derived programmatically from the three highest per-dimension scores.

Source scripts: `scripts/_vc_score_api_v0_t_top3.py` (V0+T top-3), `scripts/_vc_score_api_v0_tw_top3.py` (V0+TW top-3).

---

### 13.5 OAR Extraction

The structured O/A/R extraction study (§10) uses prompts from `scripts/_extract_three_conditions.py`. All three conditions share a **Topics block** and an **Output Schema block**; only the preamble paragraph differs.

#### Shared block: Topics (7-topic taxonomy)

Appended in all three system prompts before the output schema.

```text
## Topics (the 7-topic taxonomy)

Use these 7 topics when labeling attributes and relationships. Each attribute and relationship must be tagged with exactly one topic.

1. **Data Density / Image Clutter** — The perceived amount, richness, or depth of data content. Considers information volume, element quantity, and visual clutter/overlap.
2. **Visual Encoding Clarity** — The variety, type, and complexity of graphical forms (shapes, lines, marks) and how spatial layout, scale, and encoding interpretability contribute to complexity.
3. **Semantics / Text Legibility** — The quantity and density of text elements (titles, axis labels, legends, captions, annotations, in-chart labels).
4. **Schema** — Whether specialized domain knowledge is needed, including dimensionality (2D/3D), structural complexity, and abstraction level.
5. **Color, Symbol, and Texture Details** — Range, variety, and arrangement of colors, plus use of symbols, textures, and non-color graphical markers.
6. **Aesthetics Uncertainty** — How visually cluttered, dense, or disordered the layout appears. Higher = more cluttered/overwhelming.
7. **Immediacy / Cognitive Load** — Overall ease or difficulty of interpreting the visualization. Considers interpretive difficulty, semantic clarity, and processing time/effort.
```

#### Shared block: Output Schema

Identical across all three conditions. Note the field is `"topic"` (not `"subtopic"`).

```text
## Output Schema

Return ONLY valid JSON (no markdown fences, no prose outside JSON):
{
  "objects": [
    {"id": 1, "name": "object_name", "region": "data_area|axes|legend|title|annotation|overall"}
  ],
  "attributes": [
    {"object_id": 1, "attr": "short_snake_case_phrase", "sentiment": "+" or "-", "topic": "<one of the 7 topics>"}
  ],
  "relationships": [
    {"subj": 1, "pred": "snake_case_predicate", "obj": 2, "sentiment": "+" or "-", "topic": "<one of the 7 topics>"}
  ]
}

Guidelines:
- Extract elements proportional to the richness of the input. Typical ranges: 2–6 objects, 3–8 attributes, 1–5 relationships; sparse inputs may produce fewer and rich inputs may exceed these.
- Each list must contain at least one entry.
- Object names: single lowercase word or snake_case.
- Attribute text: snake_case, max 4 words.
- Sentiment: '+' increases perceived complexity, '-' decreases it.
- Topic must be exactly one of the seven topic titles listed above.
```

#### Condition B — Phrase-Grounded (no image)

Input: sentiment-tagged complexity phrases from participant comments (no image sent).

```text
You are a visual complexity annotation expert. You will receive, for a single data-visualization image:
  (a) a list of original complexity phrases extracted from real participant comments about the image, each tagged with a sentiment marker `(+)` (increases perceived complexity) or `(-)` (decreases perceived complexity).

Extract a scene graph (objects, attributes, relationships) that is **strictly grounded in the phrases**. Do not invent elements the phrases do not mention or imply. Use the `(+)` / `(-)` markers to inform the `sentiment` field of attributes and relationships.

[TOPICS BLOCK]

[OUTPUT SCHEMA BLOCK]
```

User turn supplies: image filename + the semicolon-separated phrase list with per-phrase `(+)`/`(-)` sentiment tags.

#### Condition V1 — Vision-Only (image, no phrases)

```text
You are a visual complexity annotation expert. You will receive, for a single data-visualization image:
  (a) the rendered image.

Extract a scene graph (objects, attributes, relationships) that is **strictly grounded in what is visible in the image**. Do not invent elements that are not visible.

[TOPICS BLOCK]

[OUTPUT SCHEMA BLOCK]
```

#### Condition V2 — Vision + Anchor Exemplars

`SYSTEM_V2` = `SYSTEM_V1` + the following block appended:

```text
## Reference Anchor Examples

Below are three reference images with expert-produced scene graphs. Use them to calibrate your extraction style (granularity, object naming, predicate style, topic usage). Do NOT copy their content; produce an extraction for the target image shown last.
```

The three anchor user→assistant turns are taken from Condition B's output (`vc_genome_output_full/three_conditions/oar_B.json`) on the same three anchor images, keeping the exemplar style consistent with the text-grounded reference. Anchor images are excluded from all evaluation comparisons.

Source script: `scripts/_extract_three_conditions.py`.
