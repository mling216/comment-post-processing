# Phrase Reduction Pipeline (BeauVis-Inspired, v2)

Revised pipeline addressing issues from the original BeauVisPhraseReduction notebook:
- **No fixed shortlist cap** — adaptive sizing so every topic (including Schema) gets sub-topic phrases
- **Criterion 5 revised** — matches BeauVis's "clearly applies to a visual representation" instead of excluding single-VisType terms
- **`secondary_topic` column** added to output
- **Stops at Step 5** for review before further processing

### Pipeline Steps
1. **Load data** — Same Google Sheets source
2. **Merge-first consolidation** — Manual synonym groups + TF-IDF cosine similarity for rare keywords
3. **Criterion-based filtering** — Revised criteria (no single-VisType exclusion)
4. **Universality scoring** — score = count × n\_vistypes, with primary & secondary topic assignment
5. **Adaptive topic-balanced selection** — Ensures all 7 topics have minimum representation
6. **Save shortlist + tracking table** — Output to `phrase_reduction_v2/`

---

## Configuration

```python
VisTypes = ['Area', 'Bar', 'Cont.-ColorPatn', 'Glyph', 'Grid', 'Line', 'Node-link', 'Point', 'Text']

topic_names = [
    'Data Density / Image Clutter',
    'Visual Encoding Clarity',
    'Semantics / Text Legibility',
    'Schema',
    'Color, Symbol, and Texture Details',
    'Aesthetics Uncertainty',
    'Immediacy / Cognitive Load'
]

topic_abbrev = {
    'Data Density / Image Clutter': 'DDIC',
    'Visual Encoding Clarity': 'VEC',
    'Semantics / Text Legibility': 'STL',
    'Schema': 'SCH',
    'Color, Symbol, and Texture Details': 'CSTD',
    'Aesthetics Uncertainty': 'AU',
    'Immediacy / Cognitive Load': 'ICL'
}
```

Output directory: `phrase_reduction_v2/`

---

## Step 0: Load Data

Load keyword counts from a Google Sheets pivot table. Build a unique phrases table by:
- Aggregating VisType involvement per keyword (which of the 9 VisTypes each phrase appears in)
- Aggregating topic involvement per keyword (which of the 7 topics each phrase belongs to)
- Summing total image counts per keyword

---

## Step 1: Merge-First Consolidation

Instead of discarding ~275 rare phrases (count ≤ 2), merge each into its closest higher-count match:

1. **Manual synonym groups** — Hand-curated groups of near-synonyms (rare + frequent)
2. **TF-IDF cosine similarity** — For remaining rare phrases, auto-merge if similarity ≥ threshold
3. **Truly unmatched** — Phrases with no good match are discarded

### Parameters

- `MIN_COUNT = 3` — Phrases with count < 3 are considered "rare"
- `SIMILARITY_THRESHOLD = 0.35` — TF-IDF cosine similarity threshold for auto-merging

### Manual Synonym Groups (16 groups)

#### Immediacy / Cognitive Load (polarity-merged: hard + easy)

**Group 1 — easy/hard to interpret:**
- more difficult to interpret/read/understand/differentiate, not interpretable/readable/understandable, hard to interpret/read/understand/differentiate, hard to interpret/read/understand, hard to compare/discern/distinguish, hard to see/tell/visualize, hard to focus/follow, hard to identify/extract/find features, more complicated, more difficult, complicated to process, hard to read, hard to describe, hard to measure/model, hard to read shape, hard to distinguish colors, harder to view data point, struggled to read, less understandable, not easily understood, less clear, less intuitive, low readability, unreadable, nothing clear, less simple, more intricate, easy to interpret/read/understand, easier to interpret/read/understand, easier to process/visualize, simple to interpret/read/understand, easier to see/tell/visualize, easy to focus/follow, easy to understand, intuitive, easy to read, easy to derive meaning, easy to identify/extract/find features, easier to compare/discern/distinguish, fast readability, more interpretable/readable/understandable, more legible, description easy to read

**Group 2 — take longer to interpret:**
- take longer to interpret, more reading/interpretation/understanding, more effort/reading/detailed analysis, attention/squinting to understand, more to read/analyze/understand, more thinking, multiple interpretation, provokes thought and understanding, require specialized knowledge, not enough knowledge to tell

#### Data Density / Image Clutter

**Group 3 — more charts/points/lines/shapes/elements:**
- more charts/points/lines/shapes/elements, multiple features/elements/graphs, many points/lines/shapes/elements, too many details/divisions, many/more data/info, few points/lines/shapes/elements, dense points/lines/shapes/elements, too many points/lines/shapes/elements, too more points/lines/shapes/elements, multiple points/lines/shapes/elements, multiple shapes, multiple interacting elements, multiple aspects, more forms, too many sections, too many subjects, too many subplots, more pixels

**Group 4 — much/little data/info:**
- much/more data/info/info spread, more detailed/things, too much data/info, fine/layered details, large dataset, diverse information, more nuance information, simple information, no data, no detail, less detailed/things, no more information, multi-year information, many measurements, mixed timeline, single variable, little data/info

**Group 5 — overlapping shapes/colors/lines (density/clutter):**
- dense/cluttered data/info, dense/cluttered layout, messy/mixed up/noisy/intermingled elements, overlapping shapes/colors/lines, dense/cluttered shape, cluttered labels/annotations, concentrated, dots scattered, scattered, scattered squares, less empty space, less negative space, negative space, convoluted composition, information overload, complex design

#### Color, Symbol, and Texture Details (merged: variety+shading, contrast+coloring)

**Group 6 — color variety/shading:**
- color variety/arrangement/distribution, too many colors, amount of / too many colors, coloring, different colors, more colors, bold colors, appealing colors, intense/too many colors, salient colors, similar colors, same colors, not simple color-wise, very different colors, colored bars/charts, colored surface, colorless, full color image, color represents sounds, color shading, black-to-white/color gradient, color scale/scheme, color hues, color intensity, color saturation, color details, color dimension, color superposition/layer, blended colors, clearer coloring, color keys, color represents groups, separate color scales, color improves readability, white color on black background, black color, blue color, green color

**Group 7 — color contrast/clarity:**
- lack of color contrast / hard to distinguish colors, unclear color meaning, ambiguous/confusing colors, more contrast, high contrast, good color contrast/separation/segmentation, low color contrast/separation/segmentation, no color difference, color differentiation, separate values by color to show differences, shapes and contrasts, unclear coloring

**Group 8 — symbols:**
- symbols, texture details, striped texture, indicating symbols, visually stimulating texture/colors, icons, nodes, symbols and shading

#### Semantics / Text Legibility

**Group 9 — amount of words/context/numbers:**
- amount of words/context/numbers, more texts/words, word/text/sentence, too many texts/words, texts/words, text, text ratio, code/text, context/text/description, too many numbers, many numbers, more numbers, ambiguous numbers, words and numbers, numbers and shapes, involves numbers, word frequency, word source, more text boxes, text on picture, text and imagery

**Group 10 — labels/axes/legends (merged: labels + legends + measurement):**
- title/axis/label, more title/axis/label, clear title/axis/label, lack of/not enough axis labels/legend/annotations/context, unclear labels, different labels, labeled axis, axis meaning, axis numbers, captions, highlighted words, label meaning, almost no explanation, axis scales, x-axis unlabeled, too much legend, legend easy to understand, lack of legend, vague legend, explanatory legend, legend on the top, with legend, no interaction, many legend categories, number of legend items, no measurement/metric/axis

**Group 11 — word rotation/small font size:**
- word rotation/small font size, different font/word sizes/structure, unclear writing, readability, easier to read, text hard to read, font, single font, small, small and simple, hierarchical text, order of words, explanation text, detailed inscription, description, word misoriented

#### Visual Encoding Clarity

**Group 12 — shapes and lines (merged: shapes + lines):**
- curves and shapes, shape variety, shape size variation, element sizes, shapes, shapes and lines, shape, overall shape, clear shape, unclear shape, shape arrangement, shape indicators, shape misunderstanding, shape subset representation, shape-color mix, shapes varied, similar shapes, less organic shapes, less uniform shapes, uneven shape, categorized shapes, colored shape, similar box sizes, mixed size, size, size variation, size decreasing, more squares than shapes, image size, length, length variation, squares, straight lines, curved lines, indistinguishable/unintuitive lines, lines, few lines, smooth lines, intersecting lines, precision of lines, line movement, line thickness variation, line quantity representation, bars as individual lines, horizontal bars, number of vertical lines, lines and scales, error bars

**Group 13 — unclear meaning/confusing:**
- clearer indication, clear indication, unclear shape meaning, unclear meaning, no clear indication, unclear meaning/confusing, indication, unclear structure, unclear what data conveys, unclear where to look/what to see, unclear relation between words, unclear filled bars, lack of meaning, different word meaning, code meaning, visualization meaning, different clarity, unclear movement, confusing connection, visual cues

**Group 14 — 2D/3D (layout/structure):**
- 2D/3D, organized/structured, spatial organization, grid layout, image layout, disposition, more uniform design, misaligned, value and graphic alignment, number of levels, number distribution, data intersection, data trend, image location, image portrayal, value representation, frame with values, values in the middle, more axes, fewer axes

#### Schema

**Group 15 — domain-specific concepts:**
- domain-specific concepts (e.g., chemical, biology, map), unfamiliar concepts/patterns, a specific technique, a specific technique, e.g., bar, pie, circle, a specific domain, abstract, no context or reference, familiar representation, technical encoding, network representation, metric representation, unknown hidden content, multivariate, unique, unique data point, nonlinearity, nonlinear measurement/metric/axis, recognizable objects, graph, plots, infomration representation, representation, context, field information, arbitrary imaging, drawing style, graphical information, factual information, information nuance expressed

#### Aesthetics Uncertainty (polarity-merged: distracting + clarity)

**Group 16 — distracting/confusing/unclear:**
- distracting/confusing/unclear, looks random/messy/lack structure, less attractive, feeling strange, visually overwhelming, visually striking, stimulus less visible, nothing visible, inconsistent, irregular, random-like movement, unable to sort, pixeled picture, different movement, visual clarity/appealing, not intuitive/simple data, clear and concise, clear connection, clear delineation

### Merge Process

1. **Manual synonym groups**: For each group, find all members present in the data. The member with the highest count becomes the **representative**. All other members map to the representative.
2. **TF-IDF auto-merge**: Remaining rare phrases (not in any manual group) are vectorized with TF-IDF (word + bigram features). Each is matched against survivor phrases (count ≥ 3) by cosine similarity. If similarity ≥ 0.35, the rare phrase merges into the best match.
3. **Truly unmatched**: Rare phrases with no match above the threshold are discarded.
4. **Count aggregation**: Merged phrases accumulate their members' counts, VisType involvement, and topic involvement into the representative.

### Representative Renaming

After merging, some representative phrases are renamed for clarity:

| Original Representative | Renamed To |
|---|---|
| easy to interpret/read/understand | easy/hard to interpret |
| much/more data/info/info spread | much/little data/info |
| title/axis/label | labels/axes/legends |
| color variety/arrangement/distribution | color variety/shading |
| lack of color contrast / hard to distinguish colors | color contrast/clarity |

---

## Step 1b: Criterion-Based Semantic Filtering (Revised)

Adapted from BeauVis Section 5.1, with **Criterion 5 revised** to match the original BeauVis wording:

| # | BeauVis Criterion | Our Adaptation |
|---|---|---|
| 1 | "Related to aesthetic pleasure" | **Related to visual complexity** — keep all (already filtered by source) |
| 2 | "Appeared ≥ 2 times" | Handled by Step 1 merge |
| 3 | "Usable in a rating scale" | **Clear directional connotation** — remove purely empty/artifact entries |
| 4 | "Easy to understand" | **Unambiguous wording** — remove data artifacts |
| 5 | "Clearly applies to a visual representation" | **REVISED: Keep domain-specific terms** — only exclude phrases with no visual dimension at all |
| 6 | "No opposite-pair redundancy" | **Remove antonym pairs** — keep higher-count member |

**Key change:** Criterion 5 no longer excludes single-VisType phrases. This preserves Schema/domain-specific phrases that were previously lost.

### Criterion Details

- **Criteria 3 & 4**: Remove data artifacts / empty entries only (e.g., `-`)
- **Criterion 5 (REVISED)**: All phrases in the data describe visual properties, so nothing is excluded beyond the artifacts above. Domain-specific and single-VisType phrases are kept.
- **Criterion 6**: Antonym-pair dedup — for each pair, keep the higher-count member and remove the lower-count one.

### Antonym Pairs

| Positive | Negative |
|---|---|
| familiar representation | unfamiliar concepts/patterns |
| organized/structured | looks random/messy/lack structure |
| high contrast | lack of color contrast / hard to distinguish colors |
| good color contrast/separation/segmentation | low color contrast/separation/segmentation |
| clearer indication | no clear indication |

---

## Step 2: Universality Scoring with Secondary Topic

$$\text{universality\_score} = \text{count} \times \text{n\_vistypes}$$

Each phrase is assigned a `primary_topic` (first listed) and `secondary_topic` (second listed, if any). Phrases appearing frequently AND across many VisTypes score highest.

### Topic Assignment

- **Primary topic**: The representative's *original* topic (from before merging expanded it). This preserves Schema's domain-specific phrases under Schema, rather than under alphabetically-first topics.
- **Secondary topic**: The first topic in the merged `topics_involved` that differs from the primary topic.

---

## Step 3: Adaptive Topic-Balanced Shortlist Selection

Key insight: each of the 55 post-merge phrases is already a **distinct semantic cluster**. Synonym groups consolidate near-duplicates, so every remaining phrase carries unique meaning. The selection strategy preserves this semantic breadth:

### Parameters

- `MIN_PER_TOPIC = 3`
- `STANDALONE_SCORE_THRESHOLD = 40`

### Selection Phases

1. **Phase 1** — Select ALL synonym group representatives (each = a distinct meaning cluster worth preserving)
2. **Phase 2** — Add standalone phrases (not in any group) above score threshold (universality\_score ≥ 40)
3. **Phase 3** — Ensure every topic has at least `MIN_PER_TOPIC` phrases; if a topic is underrepresented, add its highest-scoring remaining phrases

---

## Step 4: Verify Topic and Sub-Topic Coverage

Visual verification that all 7 topics have sub-topic phrases, including Schema. Produces:
- Summary table (topic, abbreviation, phrase count, total count, avg score)
- Horizontal bar chart grouped by primary topic, sorted by universality score

---

## Step 5: Save Final Shortlist and Reduction Tracking Table

### Outputs

- **`phrase_shortlist.csv`** — The final shortlist with columns: keyword, count, n\_vistypes, universality\_score, primary\_topic, secondary\_topic, vistypes\_involved, topics\_involved
- **`phrase_reduction_tracking.csv`** — Full tracking table showing what happened to every phrase at each step (final\_status: FINAL, merged, not selected, removed, discarded)

### Tracking Statuses

| Status | Meaning |
|---|---|
| FINAL | Selected into the shortlist |
| merged | Merged into another phrase in Step 1 |
| not selected | Survived filtering but below score threshold in Step 3 |
| removed | Removed by criterion filtering in Step 1b |
| discarded | No TF-IDF match found in Step 1, dropped entirely |

---

## VisType Profile: Phrase Distribution Across Visualization Types

For each shortlisted phrase (including all merged members), compute the total occurrence count per VisType. Produces:

1. **Heatmap** — Phrase occurrence as % of VisType images (normalized)
2. **Stacked bar chart** — Average phrase occurrence by topic, stacked per VisType
3. **Radar charts (per VisType)** — Topic profile radars for all 9 VisTypes
4. **Sub-topic radar (4 selected VisTypes)** — Individual phrase spokes colored by topic

---

## Per-Image Table

Build a table mapping each image to its VisType, topics, and the final shortlisted phrases (after merge + reduction).

### Mapping Logic

1. Every shortlisted phrase maps to itself
2. Every merged phrase maps to its representative (if the representative is in the shortlist)
3. For each image, parse all topic columns, extract raw phrases, and map them through the phrase-to-final mapping

### Output

- **`image_final_phrases.csv`** — Columns: imageName, VisType, Topics, finalPhrases

---

## Per-VisType Phrase Lineage Export

Export the full lineage chain: original phrase → human-curated representative → final sub-topic phrase.

### Lineage Resolution

1. Find merged representative (via merge\_map or phrase\_rename\_map)
2. Apply renames (some representatives were renamed after merging)
3. Check if the curated phrase is in the final shortlist

### Output

- One CSV per VisType: `phrase_lineage_{VisType}.csv`
- Combined: `phrase_lineage_all_vistypes.csv`
- Columns: VisType, topic, original\_phrase, image\_count, humanCurated\_phrase, final\_subtopic\_phrase

---

## Image–Phrase–Word Mapping

Final comprehensive export adding human-curated phrases, extracted words, and NormalizedVC.

### Word Extraction

- Uses spaCy (`en_core_web_sm`) for tokenization and POS tagging
- Uses NLTK SnowballStemmer for stemming
- Filters tokens: removes stopwords, punctuation, numbers, single characters
- Whitelists: `2d`, `3d`, `2d/3d`
- Builds a global stem-to-word mapping from all known phrases, preferring shorter lemmas
- Extracts unique representative words per image (deduplicated by stem)

### Output

- **`image_phrase_word_mapping.csv`** — Columns: imageName, imageURL, VisType, NormalizedVC, Topics, humanCuratedPhrases, finalPhrases, words\_from\_humanCurated, words\_from\_finalPhrases
