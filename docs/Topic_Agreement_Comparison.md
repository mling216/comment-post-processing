# Topic Agreement: v0 vs scoring_v4 vs Human Annotations

Comparison of two LLM-based topic extraction methods against human-annotated Topics on the 27-image subset (3 images × 9 VisTypes × 3 VC tiers).

- **v0_topics**: Direct topic selection prompt (Claude Sonnet), image-only input, selects up to 3 of 7 topics + explanation.
- **scoring_v4_topics**: VC scoring pipeline (Claude Opus), image-only input with few-shot anchors, derives top-3 topics from dimensional scores + explanation.
- **Human Topics**: Crowdsourced annotations aggregated from user study comments.

## Overall Metrics

| Metric           | v0_topics | scoring_v4 | Δ (v4 − v0) |
|------------------|----------:|----------:|-------------:|
| Sample F1        |     0.423 |     0.440 |       +0.017 |
| Macro F1         |     0.358 |     0.317 |       −0.042 |
| Jaccard (mean)   |     0.296 |     0.305 |       +0.009 |
| Macro κ          |    +0.020 |    −0.066 |       −0.086 |
| Exact match      |   3.7%    |   0.0%    |       −3.7pp |
| Partial match    |  85.2%    |  88.9%    |       +3.7pp |

## Per-Topic Cohen's Kappa

| Topic                                  | κ (v0) | κ (v4) |   Δ    | Human n | v0 n | v4 n |
|----------------------------------------|-------:|-------:|-------:|--------:|-----:|-----:|
| Data Density / Image Clutter           | +0.626 | +0.238 | −0.388 |      17 |   14 |   20 |
| Visual Encoding Clarity                | +0.036 | −0.276 | −0.311 |       7 |    7 |    5 |
| Semantics / Text Legibility            | −0.125 | −0.014 | +0.111 |       7 |   15 |    8 |
| Schema                                 | −0.326 | −0.452 | −0.126 |       6 |   15 |   14 |
| Color, Symbol, and Texture Details     | +0.129 | +0.296 | +0.167 |      12 |   16 |    7 |
| Aesthetics Uncertainty                 | −0.125 | −0.145 | −0.020 |       3 |    3 |    4 |
| Immediacy / Cognitive Load             | −0.074 | −0.107 | −0.033 |      16 |   11 |   23 |
| **Macro-average**                      | **+0.020** | **−0.066** | **−0.086** | | | |

## Key Observations

1. **Sample F1 is comparable** (~0.42–0.44) for both methods; neither achieves strong agreement with human annotations.
2. **v0 is stronger on Data Density** (κ = +0.626 vs +0.238) — the single best-performing topic. scoring_v4 over-predicts this topic (20/27 vs 17 human).
3. **scoring_v4 is stronger on Color/Symbol/Texture** (κ = +0.296 vs +0.129) — it is more conservative (7 predictions vs 16 for v0, closer to 12 human).
4. **Schema is over-predicted by both** (v0: 15, v4: 14 vs 6 human), yielding the worst per-topic kappa values (−0.326 and −0.452). LLMs systematically attribute domain-knowledge complexity that crowdsourced annotators do not emphasize.
5. **Immediacy / Cognitive Load is heavily over-predicted by scoring_v4** (23/27 vs 16 human), suggesting the scoring pipeline's dimensional structure inflates cognitive-load attribution.
6. **Neither method achieves exact match** — scoring_v4 has 0/27 exact matches; v0 achieves only 1/27.
7. **Partial overlap is high** for both (~85–89%), indicating that both methods typically identify at least one correct topic per image.

## Method Details

| Attribute        | v0_topics                        | scoring_v4_topics                          |
|------------------|----------------------------------|--------------------------------------------|
| Model            | Claude Sonnet                    | Claude Opus                                |
| Prompt style     | Direct topic selection (up to 3) | Dimensional scoring → top-3 topic derivation |
| Input            | Image only                       | Image only + 3 few-shot anchor examples    |
| Output format    | JSON: topics[] + explanation     | JSON: topics[] + explanation               |
| Cost             | Lower (Sonnet pricing)           | Higher (Opus pricing + few-shot tokens)    |

## Files

| File | Description |
|------|-------------|
| `vc_genome_output_full/ablation/subset_27_with_results_v4.csv` | Input data: 27-image subset with human Topics, v0_topics, and scoring_v4_topics |
| `scripts/compute_topic_agreement.py` | Agreement computation script (Sample F1, Jaccard, per-topic Cohen's κ) |
| `scripts/extract_ablation.py` | v0 extraction script (direct topic selection prompt, Claude Sonnet) |
| `scripts/_vc_score_api_v3.py` | scoring_v4 extraction script (dimensional VC scoring, Claude Opus + few-shot anchors) |
| `scripts/select_27_subset.py` | Subset selection script (3 images × 9 VisTypes × 3 VC tiers) |
| `scripts/merge_ablation_results.py` | Merges v0 extraction results with subset metadata |

## Inter-Method Agreement (v0 vs scoring_v4)

### Overall Metrics

| Metric           | v0 vs Human | v4 vs Human | **v0 vs v4** |
|------------------|----------:|----------:|----------:|
| Sample F1        |     0.423 |     0.440 |  **0.667** |
| Macro F1         |     0.358 |     0.317 |  **0.675** |
| Jaccard (mean)   |     0.296 |     0.305 |  **0.544** |
| Macro κ          |    +0.020 |    −0.066 | **+0.417** |
| Exact match      |   3.7%    |   0.0%    | **22.2%**  |
| Partial match    |  85.2%    |  88.9%    | **100.0%** |

### Per-Topic Cohen's Kappa (v0 vs v4)

| Topic                                  | κ (v0↔v4) | Agree | v0 n | v4 n |
|----------------------------------------|----------:|------:|-----:|-----:|
| Data Density / Image Clutter           |    +0.246 | 17/27 |   14 |   20 |
| Visual Encoding Clarity                |    +0.362 | 21/27 |    7 |    5 |
| Semantics / Text Legibility            |    +0.504 | 20/27 |   15 |    8 |
| Schema                                 |    +0.628 | 22/27 |   15 |   14 |
| Color, Symbol, and Texture Details     |    +0.388 | 18/27 |   16 |    7 |
| Aesthetics Uncertainty                 |    +0.836 | 26/27 |    3 |    4 |
| Immediacy / Cognitive Load             |    −0.049 | 11/27 |   11 |   23 |
| **Macro-average**                      | **+0.417** |       |      |      |

### Observations

1. **The two LLMs agree with each other far more than either agrees with humans** — Sample F1 jumps from ~0.43 to 0.667; Macro κ from near-zero to +0.417.
2. **Schema** (κ = +0.628) and **Aesthetics Uncertainty** (κ = +0.836) show the strongest inter-method agreement — precisely the topics where both methods most disagree with humans. This reveals a shared LLM bias: both models systematically over-attribute domain-knowledge complexity (Schema) that crowdsourced annotators do not emphasize.
3. **Immediacy / Cognitive Load** is the only topic with negative inter-method κ (−0.049): v0 predicts it for 11/27 images while v4 predicts it for 23/27, indicating the scoring pipeline's dimensional structure inflates cognitive-load attribution.
4. **100% partial match** — every image has at least one topic in common between v0 and v4.

## Conclusion

The simpler v0 prompt (Sonnet, no few-shot) performs comparably or slightly better than the scoring_v4 pipeline (Opus, few-shot anchors) when measured against human ground truth. The additional cost and complexity of the scoring pipeline does not translate to improved topic agreement. Both methods share a systematic bias toward over-predicting Schema and under-predicting topics that humans associate with surface-level visual features. The high inter-method agreement (Macro κ = +0.417) despite low human agreement suggests that LLM-based topic extraction captures a consistent but distinct perspective from crowdsourced human judgments.

## Per-VisType Agreement (Pooled κ)

With only 3 images per VisType, per-topic kappa within each group would be unreliable. Instead, we pool all 7 topic decisions within each VisType (3 images × 7 topics = 21 binary decisions) to compute a single pooled κ per group.

| VisType | v0↔H F1 | v4↔H F1 | v0↔H κ | v4↔H κ | v0↔v4 κ |
|------------------|--------:|--------:|-------:|-------:|--------:|
| Area             |   0.517 |   0.406 | +0.000 | −0.182 |  +0.806 |
| Bar              |   0.267 |   0.400 | −0.116 | +0.087 |  +0.222 |
| Cont.-ColorPatn  |   0.356 |   0.378 | +0.087 | +0.087 |  +0.417 |
| Glyph            |   0.524 |   0.524 | +0.137 | +0.137 |  +0.806 |
| Grid             |   0.633 |   0.544 | +0.493 | +0.290 |  +0.222 |
| Line             |   0.300 |   0.300 | +0.060 | +0.060 |  +0.222 |
| Node-link        |   0.362 |   0.590 | −0.085 | +0.310 |  +0.222 |
| Point            |   0.522 |   0.356 | +0.290 | +0.087 |  +0.222 |
| Text             |   0.324 |   0.457 | −0.085 | +0.113 |  +0.611 |
| **ALL**          | **0.423** | **0.440** | **+0.085** | **+0.107** | **+0.417** |

### Observations

1. **Grid** has the best v0↔Human agreement (κ = +0.493), likely because grid-based charts present unambiguous visual complexity cues (cell density, color variation) that both humans and LLMs identify consistently.
2. **Glyph and Area show very high inter-method agreement** (κ = +0.806) — the two LLMs converge on the same topics despite modest alignment with humans, reinforcing the shared-LLM-bias finding.
3. **Node-link is the one VisType where v4 clearly outperforms v0** (κ = +0.310 vs −0.085). The scoring pipeline's dimensional structure may better capture the structural complexity of network visualizations.
4. **Bar charts are hardest for v0** (κ = −0.116), while v4 achieves slight positive agreement (+0.087). Both methods struggle with this common chart type, suggesting bar chart complexity is perceived differently by humans vs LLMs.
5. **Caveat**: n = 3 per VisType means these values are indicative, not conclusive. A larger sample would be needed to confirm these patterns.
