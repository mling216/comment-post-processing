# VC-Genome: A Visual Genome–Inspired Pipeline for Visual Complexity Scene Graphs

**Date:** April 12, 2026  
**Inspiration:** Visual Genome (Krishna et al., IJCV 2017) — dense structured annotations connecting language and vision  
**Goal:** Transform participant comments from paired-image complexity comparisons into structured scene graphs that explain *why* a visualization has its perceived complexity score

---

## 1. Motivation

The current pipeline (documented in `IEEEVIS_CommentPhraseReduction_LaTeX.tex`) produces a flat taxonomy: 7 topics → 21 subtopics → ~400 curated phrases → ~670 stems. This captures *what vocabulary* participants use but not *how visual elements interact* to produce complexity.

Visual Genome's key insight is that structured representations — objects, attributes, relationships, and scene graphs — enable reasoning about images beyond what flat labels or captions provide. We adapt this idea to visualization complexity: instead of `riding(man, horse)`, we model triplets like `increases_clutter(overlapping_labels, axis_region)`.

---

## 2. Visual Genome → VC-Genome Concept Mapping

| Visual Genome Component | Per Image | VC-Genome Analog | Source |
|---|---|---|---|
| **Region descriptions** | ~50 localized phrases | Participant comments (whole-image) | Existing `rawUserComments` |
| **Objects** | ~35 named elements | VC-Objects: visual elements in the chart | Extract from comments + image |
| **Attributes** | ~26 properties | VC-Attributes: complexity-relevant properties | Extract from curated phrases |
| **Relationships** | ~21 directed triplets | VC-Relationships: how elements affect complexity | New — Claude extraction |
| **Region graphs** | per-region subgraph | VC-Region Graphs: per-comment subgraph | Assemble from above |
| **Scene graph** | 1 per image | VC-Scene Graph: unified complexity graph | Merge region graphs |
| **QA pairs** | ~17 per image | VC-QA: "Which region contributes most to complexity?" | Future extension |
| **WordNet synsets** | canonicalization | 21 named subtopics as ontology | Existing taxonomy |

### Key Differences from Visual Genome

1. **No bounding boxes.** VG grounds every object to a spatial region. We define canonical visualization regions (data area, axes, legend, annotations, whitespace) instead of pixel-level boxes.
2. **Complexity polarity.** Every attribute and relationship carries a sentiment label: `(+)` increases or `(-)` decreases perceived complexity. VG has no equivalent.
3. **LLM annotator, not crowd workers.** Claude replaces AMT workers. The Claude vs GPT evaluation (90.9% vs 81.8% accuracy on 23 pairs) validates Claude as the stronger annotator for this domain.
4. **Existing comments as input.** VG collected 50 fresh region descriptions per image. We re-parse the existing ~1,700 original phrases and ~400 human-curated phrases already collected in the study.

---

## 3. VC-Genome Data Model

### 3.1 VC-Objects

Named visual elements present in the visualization image.

| Field | Type | Example |
|---|---|---|
| `object_id` | int | 1 |
| `name` | str | `gridlines` |
| `synset` | str | `gridlines.vis.01` (canonical form) |
| `region` | str | `data_area` |
| `bounding_desc` | str | `horizontal lines spanning the plot area` |

**Canonical regions:** `data_area`, `x_axis`, `y_axis`, `legend`, `title`, `annotations`, `background`, `whitespace`, `overall`

### 3.2 VC-Attributes

Properties of VC-Objects that relate to visual complexity.

| Field | Type | Example |
|---|---|---|
| `attribute_id` | int | 1 |
| `object_id` | int | 1 (→ `gridlines`) |
| `name` | str | `dense` |
| `sentiment` | str | `(+)` |
| `subtopic` | str | `Element Quantity` |

### 3.3 VC-Relationships

Directed triplets connecting two VC-Objects or an object and an attribute.

| Field | Type | Example |
|---|---|---|
| `relationship_id` | int | 1 |
| `subject_id` | int | 1 (→ `gridlines`) |
| `predicate` | str | `overlaps_with` |
| `object_id` | int | 2 (→ `data_points`) |
| `sentiment` | str | `(+)` |
| `subtopic` | str | `Visual Clutter & Overlap` |

**Relationship types (non-exhaustive):**
- **Spatial:** `overlaps_with`, `adjacent_to`, `contains`, `surrounds`
- **Complexity-causal:** `increases_clutter_of`, `reduces_clarity_of`, `competes_with`, `reinforces`
- **Encoding:** `encodes_via`, `maps_to`, `distinguishes`
- **Cognitive:** `requires_interpretation`, `distracts_from`, `aids_reading_of`

### 3.4 VC-Region Graph

One subgraph per participant comment, containing the objects, attributes, and relationships mentioned or implied by that comment.

### 3.5 VC-Scene Graph

The union of all region graphs for a single image, with co-referenced objects merged (e.g., `grid` and `gridlines` → same node). This is the primary output artifact.

---

## 4. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT (per image)                                          │
│  • imageName, imageURL, VisType, NormalizedVC               │
│  • rawUserComments                                          │
│  • humanCuratedPhrases (with topic + sentiment labels)      │
│  • objectWords, actionWords (from existing POS extraction)  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: Structured Extraction (Claude)                    │
│                                                             │
│  For each comment/phrase cluster per image:                  │
│  1. Identify VC-Objects mentioned or implied                │
│  2. Assign VC-Attributes to each object                     │
│  3. Extract VC-Relationships between object pairs           │
│  4. Tag each element with subtopic + sentiment              │
│  5. Assign to canonical visualization region                │
│                                                             │
│  Output: list of (object, attribute, relationship) tuples   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: Canonicalization                                  │
│                                                             │
│  • Map object names to canonical vocabulary                 │
│    (using existing stem dictionary + expert-defined synsets) │
│  • Map attributes to subtopic-aligned terms                 │
│  • Normalize relationship predicates to a controlled set    │
│  • Resolve co-references across comments for same image     │
│    (IoU-equivalent: same canonical name + same region)      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: Graph Assembly                                    │
│                                                             │
│  Per comment  → VC-Region Graph                             │
│  Per image    → VC-Scene Graph (union of region graphs)     │
│                                                             │
│  Merge strategy:                                            │
│  • Objects with same canonical name + same region → merge   │
│  • Attributes accumulate (count = # comments mentioning)    │
│  • Relationships accumulate with frequency weights          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: Validation & Analysis                             │
│                                                             │
│  Graph-level metrics per image:                             │
│  • Node count (# unique objects)                            │
│  • Edge count (# relationships)                             │
│  • Attribute density (avg attributes per object)            │
│  • Complexity-positive ratio (# (+) elements / total)       │
│  • Cross-region connectivity (edges spanning regions)       │
│  • Subtopic coverage (# of 21 subtopics represented)        │
│                                                             │
│  Validation:                                                │
│  • Correlate graph metrics vs NormalizedVC                   │
│  • Compare scene graph structure across VisTypes            │
│  • Check if graph differences explain pair comparison       │
│    outcomes (the 23 evaluated pairs)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Claude Prompt Design (Stage 1)

Claude receives per image:
- The visualization image (via URL or base64)
- All raw participant comments for that image
- The human-curated phrases with topic and sentiment labels
- The 21 subtopic definitions as the target ontology

Claude is asked to produce a JSON structure:

```json
{
  "imageName": "InfoVisJ.1234.5",
  "vc_objects": [
    {"id": 1, "name": "bar", "region": "data_area", "count_in_comments": 8},
    {"id": 2, "name": "axis_label", "region": "x_axis", "count_in_comments": 3}
  ],
  "vc_attributes": [
    {"object_id": 1, "attribute": "densely_packed", "sentiment": "+", "subtopic": "Element Quantity"},
    {"object_id": 2, "attribute": "overlapping", "sentiment": "+", "subtopic": "Visual Clutter & Overlap"}
  ],
  "vc_relationships": [
    {
      "subject_id": 1, "predicate": "overlaps_with", "object_id": 2,
      "sentiment": "+", "subtopic": "Visual Clutter & Overlap"
    }
  ]
}
```

---

## 6. Proof-of-Concept Scope

| Parameter | Value |
|---|---|
| **Images** | 20–30 (sampled across VisTypes and NormalizedVC range) |
| **Annotator** | Claude (via API) |
| **Input data** | `image_compiled_phrases.csv` |
| **Ontology** | 21 subtopics from `phrase_shortlist.csv` |
| **Output** | Per-image JSON scene graphs + summary CSV |
| **Validation** | Pearson/Spearman correlation of graph metrics vs NormalizedVC |

### Success Criteria

1. Graph metrics show statistically significant correlation (p < 0.05) with NormalizedVC
2. Scene graphs for the 23 evaluated pairs produce complexity orderings consistent with human ground truth at ≥ 85% accuracy
3. Qualitative review confirms that scene graphs capture meaningful structural differences between high-VC and low-VC images

---

## 7. Relation to Existing Pipeline

The VC-Genome pipeline is **complementary**, not a replacement. It adds a structural layer on top of the existing phrase taxonomy:

| Existing Output | VC-Genome Extension |
|---|---|
| 21 named subtopics | Ontology for canonicalization |
| ~400 curated phrases | Input for relationship extraction |
| Stem dictionary (~670 words) | Vocabulary for object/attribute naming |
| VC-Attribute checklist (19 items) | Validation target — do scene graphs cover all 19 attributes? |
| Claude pair comparisons (23 pairs) | Validation dataset — do scene graph diffs predict pair outcomes? |

---

## 8. Open Questions

1. **Region granularity:** Are the 9 canonical regions sufficient, or do we need finer-grained spatial decomposition (e.g., individual axis tick marks)?
2. **Relationship vocabulary:** Should we define a closed set of predicates, or let Claude propose predicates and then canonicalize?
3. **Graph comparison metrics:** What graph similarity/distance measures best capture "complexity difference" between two images?
4. **Scaling:** Can the prototype scale to all 520 images within API cost constraints?
5. **Human validation:** Should a subset of scene graphs be reviewed by visualization experts before analysis?
