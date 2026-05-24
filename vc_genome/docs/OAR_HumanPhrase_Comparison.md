# OAR-B vs. Human-Curated Vocabulary: Comparison Methodology

**Notebook:** `vc_genome/notebooks/OAR_HumanPhrase_Comparison.ipynb`  
**Date:** May 2026

---

## 1. Motivation

The OAR pipeline (Object–Attribute–Relationship) uses an LLM to extract structured scene
graphs from visualization images, constrained by a human-designed synset of ~300 object
categories. The human curation track independently produced two artifacts from the same
images via participant comments:

- **finalCuratedPhrases.csv** — 404 manually curated multi-word phrases, each tagged
  with topic, sentiment, and visualization-type coverage.
- **finalDictionary.csv** — 689 stemmed vocabulary terms derived from those phrases,
  with per-topic image-count weights and POS tags.

These two tracks were developed independently. The comparison asks:
*Do the LLM's structured outputs express the same vocabulary and concepts that human
annotators chose?*

---

## 2. Data Sources

| Artifact | Path | Rows | Format |
|---|---|---|---|
| OAR-B (66-image eval set) | `vc_genome_output_full/three_conditions/oar_B.json` | 66 images | JSON |
| OAR-B (510-image full set) | `vc_genome_output_full/three_conditions/oar_B_510.json` | 510 images | JSON |
| Human phrases | `comment_process/finalCuratedPhrases.csv` | 404 phrases | CSV |
| Human dictionary | `comment_process/finalDictionary.csv` | 689 stems | CSV |

**OAR JSON structure per image:**
```json
{
  "objects":       [{"id": ..., "name": "bar_chart", "region": ...}],
  "attributes":    [{"object_id": ..., "attr": "unclear_color_meaning", "sentiment": "+", "topic": "..."}],
  "relationships": [{"subj": ..., "pred": "increases_cognitive_load_of", "obj": ..., "sentiment": "+", "topic": "..."}]
}
```

---

## 3. Comparison Method: Token-Level Overlap

### 3.1 Why token-based matching?

OAR terms are stored as `snake_case` compounds (e.g. `unclear_color_value_meaning`,
`increases_cognitive_load`). The human dictionary stores individual **stems**
(e.g. `unclear`, `color`, `meaning`). Neither representation is a free-form sentence,
so the natural unit of comparison is the **word token**: split OAR compounds on `_`,
filter stopwords, and intersect with dictionary stems.

This is conceptually equivalent to **ROUGE-1** (Lin 2004) and **bag-of-words overlap**
used in NLP summarization evaluation and scene graph assessment (Krishna et al.,
Visual Genome, 2017): structured output tokens are compared against a reference
vocabulary via set intersection.

### 3.2 Tokenization pipeline

```
OAR term  →  split on '_'  →  lowercase  →  remove stopwords  →  flat token set
"unclear_color_value_meaning"  →  ["unclear", "color", "value", "meaning"]
```

Human phrases undergo the same split (on spaces/punctuation) before coverage is
computed.

### 3.3 What this measures — and what it does not

| Captured | Not captured |
|---|---|
| Exact vocabulary reuse | Synonymy (`cluttered` ≠ `messy`) |
| Shared conceptual building blocks | Hypernymy (`color` ≠ `hue`/`saturation`) |
| Whether LLM draws from human lexicon | Paraphrase (`hard to read` ≠ `low readability`) |

Results are therefore **lower bounds**. Lemmatization or semantic similarity (e.g.
spaCy word vectors, cosine distance) would raise both overlap figures.

---

## 4. Analyses and Results (510-image OAR-B set)

### 4.1 Vocabulary size

| Set | Count |
|---|---|
| OAR unique object terms | 368 |
| OAR unique attribute terms | 1510 |
| OAR unique relationship predicates | 451 |
| **OAR distinct tokens** (after split + stopword removal) | **1186** |
| Human dictionary stems | 689 |
| Human curated phrases | 404 |

### 4.2 Dictionary ↔ OAR token overlap

| Metric | Value |
|---|---|
| Stems found in OAR tokens (exact match) | **453 / 689 (65.8%)** |
| Stems absent from OAR | 235 |
| OAR tokens absent from dictionary | 733 |

**Top absent dictionary stems** (by image frequency): `datum`, `confuse`, `know`,
`mean`, `write`, `need`, `go`, `show` — mostly generic cognitive/action verbs that the
LLM reformulated into structured attribute phrases.

**Top OAR-only tokens**: `increases`, `colors`, `labels`, `visualization`, `load`,
`elements`, `contributes`, `encoding` — largely inflected forms and domain composites
not present as stems in the human dictionary.

*Output:* `figures/oar_vs_dict_overlap.png`, `figures/dict_stems_absent_from_oar.csv`

### 4.3 Topic distribution alignment

OAR attributes and relationships are tagged with one of 7 topics. Human phrases also
carry a topic label. Comparison is by normalised frequency:

| Topic | OAR% | Phrase% | Dict coverage |
|---|---|---|---|
| Data Density / Image Clutter | 17% | 18% | 1949 |
| Visual Encoding Clarity | 15% | 16% | 1773 |
| Semantics / Text Legibility | 13% | 16% | 1818 |
| Schema | 6% | 8% | 1254 |
| Color, Symbol, and Texture Details | 14% | 16% | 2105 |
| Aesthetics Uncertainty | 11% | 3% | 719 |
| **Immediacy / Cognitive Load** | **23%** | **19%** | 2170 |

Key observations:
- *Immediacy / Cognitive Load* is the dominant OAR topic; slightly over-represented
  vs. human phrases.
- *Aesthetics Uncertainty* is markedly under-represented in human phrases (3%) compared
  to OAR (11%) and the dictionary (6%) — the LLM captures aesthetic uncertainty more
  explicitly than human annotators phrased it.
- *Schema* is under-represented in OAR (6%) vs. human phrases (8%) and dictionary (11%).

*Output:* `figures/oar_vs_human_topic_align.png`

### 4.4 Phrase coverage by OAR vocabulary

Each human phrase is tokenized and its tokens are checked against `all_oar_tokens`:

| Metric | Value |
|---|---|
| Mean OAR token coverage per phrase | **93.6%** |
| Mean dictionary token coverage per phrase | 57.3% |
| Phrases **fully** covered by OAR tokens (100%) | 346 / 404 (86%) |
| Phrases with **zero** OAR coverage | 6 |

Phrases with zero OAR coverage: `nonlinearity`, `more pixels`, `disposition`, `dates`,
`misaligned`, `no interaction` — all rare phrases (≤2 images each).

*Output:* `figures/oar_phrase_coverage.png`, `figures/oar_vs_human_phrase_gaps.csv`

### 4.5 Reverse coverage: OAR attributes → human phrases

For each OAR attribute term, its tokens are matched against phrase tokens to find which
human phrases it "touches":

| Metric | Value |
|---|---|
| OAR attribute terms matched to ≥1 phrase | **1389 / 1510 (92%)** |
| OAR attribute terms with no phrase match | 121 |

The 121 unmatched attribute terms represent concepts the LLM introduced that have no
direct human phrase equivalent, e.g.: `geo_map_format`, `temporal_comparison_required`,
`circular_chart_form`, `non_linear_pattern`, `never_seen_before`, `has_internal_logic`.

*Output:* `figures/oar_attr_phrase_match.csv`

### 4.6 Triple-to-natural-language reconstruction

OAR relationship triples from the 66-image eval set are rendered as subject–predicate–
object sentences to assess their expressiveness as natural language:

```
dots increases clutter in chart               [Data Density / Image Clutter, +]
color scheme enhances readability of chart    [Aesthetics Uncertainty, −]
text labels increases cognitive load of chart [Immediacy / Cognitive Load, +]
```

165 triples across 66 images; 92% of relationship predicates map to ≥1 human phrase.

---

## 5. Summary Interpretation

The OAR-B extraction and the human-curated vocabulary are **highly aligned at the
token level**: 93.6% of human phrase tokens appear in the OAR lexicon, and 92% of OAR
attribute terms link back to at least one human phrase. This validates that the LLM
respected the human-designed synset and produced outputs in the same conceptual space.

The main divergences are:
1. **Inflection / stemming gap** — OAR uses unstemmed inflected forms (`increases`,
   `labels`) that don't match stemmed dictionary entries (`increase`, `label`). A
   lemmatization pass would close most of the 65.8% → ~85% gap.
2. **OAR-novel concepts** — 121 attribute terms and several hundred tokens describe
   things the human curators did not phrase explicitly (e.g. map geography, circular
   layout, temporal structure). These are genuine LLM additions.
3. **Human-novel vocabulary** — 235 dictionary stems (mostly generic cognitive verbs:
   `confuse`, `know`, `mean`) were not reproduced by the LLM, which tends to
   reformulate these into more specific structured attributes.
4. **Topic skew** — OAR over-indexes on *Aesthetics Uncertainty* and *Immediacy /
   Cognitive Load* relative to the human phrase distribution.

---

## 6. Limitations and Next Steps

- All overlap is **exact string match** — lemmatization or cosine-similarity on word
  embeddings would yield a more complete picture.
- OAR-V1 and OAR-V2 conditions are not yet compared against human vocabulary.
- The topic alignment comparison uses raw counts; a normalized per-image rate would
  be more informative.
