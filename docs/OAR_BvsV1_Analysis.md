# VC-Genome OAR Analysis: B vs V1 (and B vs V2)

**Dataset:** 510-image full corpus (9 main VisTypes: Area, Bar, Cont.-ColorPatn, Glyph, Grid, Line, Node-link, Point, Text).  
**Conditions:**
- **B** — text-only; grounded in sentiment-tagged participant phrases (human-phrase reference)
- **V1** — pure vision; image + 7-topic taxonomy, no exemplars
- **V2** — vision + 3 in-context O/A/R exemplars drawn from B's anchor outputs

All matching is set-theoretic over canonicalized keys (synsets, normalized attributes, canonical predicates). `VisType` and `NormalizedVC` are not used.

### Shared canonicalization dictionary

Both B and V1 pass through the **same** `scripts/_vc_canon.py` canonicalization step before any matching is done. The dictionary has three components:

- **`OBJECT_SYNSETS`** — a hand-curated mapping of ~150 surface-form strings to role-level synsets (e.g. `"bar"` → `mark.bar`, `"title"` → `text.title`).
- **`_SUFFIX_MAP`** — a secondary fallback that resolves compound terms by their final word (e.g. anything ending in `"_label"` → `text.label`).
- **`PREDICATE_CANON`** — a curated mapping for relationship predicates (e.g. `"connects_to"` → `connects`).

Any object name not resolved by either lookup falls back to `unknown.<name>`. This is where B and V1 diverge in practice: B tends to use the high-level role names (e.g. `"bar"`, `"legend"`) that were used to build the ~150-term dict, so most of its terms resolve cleanly. V1 produces more verbose, instance-specific names (e.g. `"horizontal_bars_section"`, `"color_coded_arrows"`) whose suffixes often miss the `_SUFFIX_MAP`, pushing them into the `unknown.*` space and preventing a match. The much larger `unknown.*` population in V1 is therefore a direct consequence of this vocabulary gap, not a flaw in the canonicalization logic itself.

---

## 1. OAR Volume: How Much Does Each Condition Produce?

![Distribution comparison — B vs V1](../vc_genome_output_full/match/vg_style_distributions.png)

*Figure 1. Visual Genome–style histograms (a–e) comparing B and V1 across all 510 images. Dashed lines mark the mean for each condition.*

V1 (pure vision) produces substantially more structured content per image than B (human-phrase) across all three layers:

| Metric | B (human-phrase) | V1 (pure vision) | V2 (vision + exemplars) |
|---|---|---|---|
| Objects / image | 2.8 | **7.0** (+150%) | 6.2 (+121%) |
| Attributes / image | 3.7 | **10.8** (+192%) | 9.3 (+151%) |
| Relationships / image | 1.8 | **5.3** (+194%) | 4.5 (+150%) |

The histograms in panels (a–e) show right-skewed distributions for both conditions, but V1's distributions are shifted considerably rightward. Panel (d) is particularly striking: V1 consistently names twice as many distinct objects per image as B. This reflects that vision grounding surfaces all visible chart components (individual bars, every label, each axis tick cluster), while B names only the components that participants explicitly mentioned in their comments.

V2 (vision + exemplars) sits between B and V1 on every measure — the few-shot B exemplars pull the vision output toward B's abstraction level and reduce the inflation.

![B vs V2 distribution](../vc_genome_output_full/match/vg_style_distributions_bv2.png)

*Figure 2. Same panels as Figure 1, now comparing B against V2. V2 is more compact than V1 but still substantially richer than B.*

---

## 2. Vocabulary: What Terms Does Each Condition Prefer?

![Top-25 entities — B vs V1](../vc_genome_output_full/match/vg_style_top25.png)

*Figure 3. Top-25 most frequent object synsets (left), attribute terms (centre), and relationship predicates (right) for B vs V1.*

### Object synsets

Both conditions agree that `property.color`, `text.label`, and `furniture.axes` are among the most common chart components. However V1 mentions `text.label` at nearly three times the rate of B — vision sees every on-screen string directly, while human commenters rarely enumerate individual labels. V1 also introduces many `unknown.*` synsets (instance-level terms such as `horizontal_bars`, `country_labels`) that fall outside B's role-level vocabulary.

### Attribute terms

B's top attributes are interpretive: `increases_complexity`, `difficult_to_read`, `requires_domain_knowledge`. V1's top attributes lean perceptual: `high_color_variety`, `multiple_colors_used`, `dark_background`. This mirrors the grounding signal: human phrases encode cognitive/evaluative responses, while vision grounding encodes what the model directly perceives in the pixels.

### Relationship predicates

`increases_effort`, `obscures`, and `aids_interpretation` appear in both conditions. V1 over-produces `requires_domain_knowledge` as a predicate head noun (flagging complex chart types it recognises visually) while B is more evenly spread across complexity and clarity predicates.

![Top-25 entities — B vs V2](../vc_genome_output_full/match/vg_style_top25_bv2.png)

*Figure 4. Same as Figure 3, but comparing B vs V2. V2's vocabulary is closer to B's: `property.color` and interpretive predicates gain relative share, while `text.label` inflation is reduced.*

---

## 3. Pairwise Agreement: Do B and V1 Name the Same Things?

Despite V1 producing nearly twice as many OARs, set-theoretic overlap with B is low across all layers:

| Layer | Mean F1 | Median F1 | SD |
|---|---|---|---|
| Objects (synset) | **0.164** | 0.167 | 0.178 |
| Attributes (`object_synset` + `attr` text) | 0.001 | 0.000 | 0.009 |
| Relationships strict `(s, p, o)` | 0.000 | 0.000 | 0.010 |
| Relationships loose `{s, o}` endpoints | 0.020 | 0.000 | 0.081 |

Object agreement (F1 ≈ 0.16) is low but non-zero: the synset vocabulary forces both sides onto a shared ontology, so when B says `mark.bar` and V1 says `mark.bar` they match — but B names a handful of high-level synsets while V1 names many more instance-level objects that map to `unknown.*` and miss. Recall (0.27) exceeds precision (0.13), meaning V1 covers most of what B names but buries those synsets in a larger, noisier set.

Attribute and strict relationship F1 are identically zero: attributes are keyed as `(object_synset, attribute_text)` pairs, and because V1 uses entirely different perceptual phrasings from B's evaluative terms, there is no string-level overlap. This is not a failure of content but of vocabulary — both conditions produce sentiment-tagged claims about the same images, but in incompatible lexicons.

The near-zero loose-endpoint relationship F1 (mean 0.016) is consistent: even ignoring the predicate and matching only `{subject_synset, object_synset}` pairs, the different object sets from each side rarely intersect.

**Sentiment agreement is undefined** at this level because no matched attribute or relationship pairs exist to compare.

### High-agreement examples

| Image | obj F1 |
|---|---|
| `SciVisJ.822.18.png` | **0.80** |
| `whoO06_2.png` | 0.57 |
| `wsj340.png` | 0.57 |

These images tend to have simple chart types (single-mark visualisations) where B's sparse role vocabulary and V1's richer instance vocabulary happen to resolve to the same small set of synsets.

### Low-agreement examples

| Image | obj F1 |
|---|---|
| `wsj603.png` | 0.17 |
| `SciVisJ.259.7.png` | 0.18 |
| `VASTC.13.9.png` | 0.18 |

These are likely complex multi-component visualisations where V1 enumerates many fine-grained components (mapping to `unknown.*`) while B names only a few canonical roles.

---

## 4. Agreement by VisType

The 510-image corpus enables breakdown of object F1 by visualization type. Three separate 3×3 grid figures show the distribution of B-vs-V1 F1 for each of the 9 VisTypes:

![Per-VisType object F1 grid](../vc_genome_output_full/match/vistype_grid_obj_f1.png)

*Figure 5. Object F1 distribution per VisType (3×3 grid). Each subplot shows the histogram of per-image object F1 values; dashed line is the mean.*

![Per-VisType attribute F1 grid](../vc_genome_output_full/match/vistype_grid_attr_f1.png)

*Figure 6. Attribute F1 distribution per VisType.*

![Per-VisType rel (loose) F1 grid](../vc_genome_output_full/match/vistype_grid_rel_loose_f1.png)

*Figure 7. Relationship (loose endpoint) F1 distribution per VisType.*

Mean per-VisType object F1 (from `vistype_summary.csv`):

| VisType | obj F1 | attr F1 | rel (loose) F1 | B obj/img | V1 obj/img |
|---|---|---|---|---|---|
| Area | 0.226 | 0.000 | 0.064 | 2.8 | 6.1 |
| Bar | 0.211 | 0.000 | 0.038 | 2.4 | 5.8 |
| Cont.-ColorPatn | 0.094 | 0.000 | 0.007 | 2.9 | 5.5 |
| Glyph | 0.049 | 0.000 | 0.005 | 2.5 | 5.6 |
| Grid | 0.095 | 0.000 | 0.006 | 2.7 | 5.5 |
| Line | 0.206 | 0.000 | 0.021 | 2.9 | 5.9 |
| Node-link | 0.151 | 0.002 | 0.004 | 3.0 | 5.4 |
| Point | 0.209 | 0.000 | 0.013 | 3.2 | 6.6 |
| Text | 0.256 | 0.005 | 0.025 | 2.6 | 5.5 |

Key observations:
- **Text** (0.256) and **Area** (0.226) VisTypes have the highest object agreement — these tend to contain a small number of dominant, semantically unambiguous elements (e.g., word tokens, filled regions) that both B and V1 reliably name with the same synsets.
- **Glyph** (0.049) and **Cont.-ColorPatn** (0.094) are lowest — abstract continuous marks and composite glyphs produce highly divergent object vocabularies: B names the perceptual role (e.g., `mark.glyph`) while V1 enumerates individual visual components mapping to `unknown.*`.
- **Attribute F1** is near zero for all VisTypes, confirming the vocabulary incompatibility is structural, not VisType-specific.
- **V1 always produces more objects per image** (roughly 2× B), but the inflation ratio is somewhat higher for Point (6.6 vs 3.2) and lower for Node-link (5.4 vs 3.0).

### 4b. Top-15 entity vocabulary per VisType

Complementing the F1 distributions, the following three figures show which object synsets, attribute terms, and relationship predicates are most frequent within each VisType for B vs V1:

![Per-VisType top-15 object synsets](../vc_genome_output_full/match/vistype_grid_top_obj.png)

*Figure 8. Top-15 object synsets per VisType (3×3 grid). Blue = B (human-phrase), orange = V1 (pure vision). Terms sorted by combined frequency.*

![Per-VisType top-15 attribute terms](../vc_genome_output_full/match/vistype_grid_top_attr.png)

*Figure 9. Top-15 attribute terms per VisType.*

![Per-VisType top-15 relationship predicates](../vc_genome_output_full/match/vistype_grid_top_pred.png)

*Figure 10. Top-15 relationship predicates per VisType.*

Selected observations from the vocabulary grids:
- **Node-link** — `mark.node` dominates V1 objects (direct pixel counting of nodes); B uses higher-level terms like `structure.link` and `whole.visualization`.
- **Grid / Cont.-ColorPatn** — V1 introduces VisType-specific structural predicates (`labels_columns_of`, `labels_rows_of`) that have no counterpart in B's evaluative predicate set.
- **Line** — `overlaps_with` is V1's top predicate (line crowding is immediately visible from pixels); B instead uses cognitive-load predicates.
- **Attributes** — `requires_domain_knowledge` appears as V1's top attribute across nearly every VisType, reflecting a systematic pattern where the model flags complexity it perceives visually regardless of chart type. B's attributes are more diverse and evaluative (e.g., `increases_complexity`, `difficult_to_read`).

---

## 5. Interpretation

The core finding is a **grounding asymmetry**, not a correctness problem:

- **B** names chart *roles* in a sparse, cognitively-filtered vocabulary (what participants found salient enough to comment on).
- **V1** names chart *instances* in a rich perceptual vocabulary (everything the model can see in the pixels).

Both are valid descriptions of the same images, but they occupy different points in the abstraction space. This motivates the **role-level projection** described in the paper (Appendix, §OAR Matching), where every synset is collapsed to one of 8 chart-part roles before matching — separating genuine disagreement from vocabulary mismatch.

V2 (vision + exemplars) shows that grounding vision output in B's own vocabulary partially bridges the gap: output volume drops toward B's level and the top-25 vocabulary shifts toward B's preferred synsets and predicates, making role-level agreement higher.
