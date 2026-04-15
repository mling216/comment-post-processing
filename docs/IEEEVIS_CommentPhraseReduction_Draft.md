# Draft Section (IEEE VIS): From Participant Comments to Topic Subtopics

## Comment Processing and Phrase Reduction Pipeline

To analyze participant comments from paired-image complexity comparisons, we designed a multi-stage human-in-the-loop pipeline that combines LLM assistance with expert curation. The pipeline converts free-form textual comments into a compact set of interpretable subtopics used to characterize perceived image complexity.

### Stage 1: Topic Framework Construction

We first prompted an LLM to summarize high-level themes in the raw participant comments. Visualization experts then reviewed and revised these candidate themes to form seven final analysis topics:

1. Data Density / Image Clutter
2. Visual Encoding Clarity
3. Semantics / Text Legibility
4. Schema
5. Color, Symbol, and Texture Details
6. Aesthetics Uncertainty
7. Immediacy / Cognitive Load

This expert revision step ensured topic definitions reflected visualization semantics and reduced ambiguity in downstream phrase assignment.

### Stage 2: Phrase Extraction, Sentiment Labeling, and Topic Assignment

Next, we prompted an LLM to extract short phrases from participant comments, assign each phrase to one or more topics, and label phrase-level complexity sentiment. In this labeling scheme, `(+)` indicates higher perceived complexity and `(-)` indicates lower perceived complexity. For example, from the comment “Curves and graphs are always hard to understand for me,” the extracted phrase was mapped to Immediacy / Cognitive Load and labeled with positive complexity sentiment.

Because LLM outputs often included long or overlapping expressions, we manually curated these assignments by:

1. Resolving multi-topic assignments to the most relevant topic.
2. Shortening overly long expressions to compact units (e.g., “hard to understand”).

This process produced 1,700+ original phrases across the seven topics, each with topic assignment and complexity polarity labels.

### Stage 3: Progressive Consolidation (Original -> AI-Curated -> Human-Curated)

We then used an LLM to group semantically similar original phrases into a smaller AI-curated set (about 500 phrases). Visualization experts manually reviewed these AI-curated phrases and further merged/revised them to produce about 400 HumanCurated phrases.

Although substantially reduced, this set still contained redundancy and cross-topic overlap. To derive concise subtopics per main topic, we adopted a BeauVis-inspired reduction process [\cite{he2022beauvis}] implemented in our PhraseReductionPipeline notebook.

## BeauVis-Inspired Subtopic Reduction Procedure

Starting from the ~400 HumanCurated phrases, the notebook applies six key steps to obtain the final subtopic shortlist (20 phrases).

### Step 0: Aggregate Phrase Statistics

We aggregate phrase usage statistics from the annotation table, including:

1. Phrase frequency across all images (`count`).
2. Number of visualization types in which a phrase appears (`n_vistypes`).
3. Topic and VisType coverage fields for later diagnostics.

These statistics provide the basis for consolidation and ranking.

### Step 1: Merge-First Consolidation

Instead of discarding rare phrases early, we first merge them into semantically close representatives.

1. Manual synonym-group merging.
- Experts define synonym groups spanning lexical variants and closely related expressions.
- Within each group, the most frequent phrase becomes the representative label.

2. TF-IDF similarity matching for remaining rare phrases.
- Phrases below a minimum count threshold are compared to higher-frequency survivors using TF-IDF cosine similarity.
- Each unmatched rare phrase is merged into its closest survivor representative.

This merge-first strategy preserves low-frequency but meaningful concepts by absorbing them into stable representatives rather than dropping them prematurely.

### Step 1b: Criterion-Based Semantic Filtering (BeauVis-Inspired, Revised)

After consolidation, we apply criterion-based filtering adapted from BeauVis [\cite{he2022beauvis}]. The key methodological revision is Criterion 5:

1. We keep domain-specific and single-VisType phrases if they still clearly describe visual properties.
2. We remove only non-informative artifacts (e.g., empty placeholders) and lower-frequency members of explicit antonym pairs.

This revision was important to preserve Schema-related language that would otherwise be disproportionately removed.

### Step 2: Universality Scoring with Topic Assignment

Each retained phrase receives a universality score:

$$
\text{universality\_score} = \text{count} \times \text{n\_vistypes}.
$$

The score favors phrases that are both frequent and broadly observed across visualization types. We retain both primary and secondary topic assignments to capture cross-topic semantics while preserving each phrase's canonical topic origin.

### Step 3: Adaptive Topic-Balanced Selection

To construct an interpretable final shortlist while maintaining topic coverage, we use a three-phase adaptive selection policy:

1. Include all synonym-group representatives (to preserve each consolidated semantic cluster).
2. Add standalone high-scoring phrases (above a fixed universality threshold).
3. Enforce per-topic minimum coverage so all seven topics contribute subtopics.

In our final configuration, this process yields 20 subtopics distributed across all seven topics.

### Step 4: Coverage Verification and Diagnostics

We generate topic-level summaries and visual diagnostics (e.g., score-ranked plots by topic) to verify that:

1. Each topic has sufficient subtopic representation.
2. Schema remains represented after reduction.
3. Selected phrases are not dominated by a narrow subset of visualization types.

### Step 5: Reproducible Outputs and Lineage Tracking

Finally, the pipeline exports:

1. A final shortlist table with phrase metadata (count, VisType coverage, score, primary/secondary topics).
2. A full tracking table that records whether each phrase was merged, filtered, retained, or not selected.

This lineage table supports auditability and enables iterative refinement of the phrase taxonomy.

## Why This Pipeline Was Needed

A direct manual reduction from ~400 phrases to a small subtopic set is difficult to reproduce and prone to inconsistency. The above workflow formalizes reduction as a sequence of transparent operations: merge, filter, score, and topic-balanced select. This preserves semantic breadth from participant comments while producing a compact and analyzable subtopic vocabulary suitable for downstream quantitative and qualitative analysis in IEEE VIS-style studies.

## Lexical Dictionary Construction

In addition to phrase-level subtopic reduction, we built a lexical dictionary from the original participant-comment phrases. We parsed the phrases to derive token-level linguistic features and extracted keyword stems, part-of-speech (POS) tags, and phrase-level action/object word components for each topic-linked phrase.

Implementation in the notebook uses spaCy POS analysis together with Snowball stemming to normalize lexical variants, and then consolidates repeated forms through stem-based deduplication. This produces a compact vocabulary that is still aligned with the seven-topic framework and the underlying phrase lineage.

The resulting dictionary contains about 670 unique words across the seven topics, and is used as a complementary representation of participant language alongside the 20 final phrase subtopics.
