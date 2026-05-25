# OAR Extraction — curate_dict Condition

## Overview

The **curate_dict** condition extracts Object–Attribute–Relationship (OAR) scene graphs from participant complexity phrases, using human-curated data to constrain what the LLM is allowed to name and tag. It is designed as a principled improvement over the B condition: instead of letting the model freely invent object names or subtopic labels, both are grounded in per-image participant data.

---

## Prompt Design

### System Prompt

The system prompt tells the model to act as a visual complexity annotation expert and to extract a scene graph that is **strictly grounded in the input phrases** — it must not invent objects or relationships the phrases do not mention or imply.

Three constraint sections are declared:

| Section | Role |
|---|---|
| **Preferred Object Vocabulary** | Instructs the model to use participant-derived object terms when they fit; allows fallback to specific snake_case names if none fit |
| **Topics** | Declares the 7 fixed topic categories for tagging |
| **Subtopics** | Instructs the model to tag using only the subtopics listed in the user message for that specific image |

The output schema is defined inline in the system prompt (see below).

### User Message (per image)

Built by `format_user_message()` from `_extract_oar_curate_dict.py`. Contains four blocks:

```
Image: <imageName>

Preferred object vocabulary (from participant descriptions of this image):
<comma-separated objectWords>

Subtopics for this image (use exactly these for tagging):
- <SubTopic 1>
- <SubTopic 2>
...

Complexity phrases from participant comments:
- <phrase 1>
- <phrase 2>
...

Extract the scene graph grounded in these phrases.
```

Sentiment markers (`(+)` / `(-)`) are stripped from phrases before sending. Neither the input nor the output schema includes a `sentiment` field.

---

## Closed-Form vs. Open-Form

| Field | Form | Reason |
|---|---|---|
| **Object `name`** | **Soft-closed** (preferred vocabulary) | Object names come from `objectWords` — noun stems extracted from participant descriptions of that specific image. The model is instructed to prefer these terms but may use its own snake_case name if none fit. This grounds object identity in real participant language while allowing graceful fallback. |
| **Object `region`** | **Closed** (9 fixed values) | Region is one of: `data_area`, `x_axis`, `y_axis`, `legend`, `title`, `annotation`, `colorbar`, `background`, `overall`. These cover all structural zones of a data visualisation and are defined in the output schema. |
| **`topic`** | **Closed** (7 fixed labels) | Topics are the 7 pre-defined complexity dimensions shared across all conditions. Defined in the system prompt. |
| **`subtopic`** | **Closed** (per-image list) | Subtopics are the specific subtopics that participants associated with this image, drawn from `SubTopics` in the data CSV. The model must choose from this list only — not from the full 19-subtopic ontology. This makes tagging image-specific and grounded. |
| **Attribute `attr`** | **Open** (free snake_case) | Attribute text is free-form snake_case, max 4 words. The model derives it from the phrase content. No closed vocabulary exists for attribute descriptions, as they capture diverse qualitative observations. |
| **Relationship `pred`** | **Open** (free snake_case) | Predicate text is free-form snake_case derived from the phrases. Relationship predicates are inherently compositional and too varied to enumerate in advance. |

---

## Model & API Settings

| Parameter | Value |
|---|---|
| Model | `claude-sonnet-4-6` |
| Temperature | `0.0` |
| Max tokens | `2048` |
| Concurrency | `5` async workers |
| System prompt caching | Enabled (`ephemeral` cache control) |

---

## Input Tables

| File | Location | Columns used |
|---|---|---|
| `image_compiled_phrases.csv` | `phrase_reduction_v2/outputs/` | `imageName`, `imageURL`, `VisType`, `NormalizedVC`, `originalPhrases`, `Topics`, `SubTopics`, `objectWords`, `actionWords` |

**Sampling (9-image mode):** One image is sampled per 0.1 `NormalizedVC` bin (bins 0–0.1, 0.1–0.2, …, 0.9–1.0), restricted to the 9 main VisTypes (`Area`, `Bar`, `Cont.-ColorPatn`, `Glyph`, `Grid`, `Line`, `Node-link`, `Point`, `Text`). Within each bin, rows where `objectWords` or `SubTopics` are empty are skipped; a fallback to any row in the bin is used only if all are invalid.

---

## Code Files

| Script | Location | Purpose |
|---|---|---|
| `_extract_oar_curate_dict.py` | `vc_genome/scripts/` | LLM extraction — builds prompts, calls API, saves JSON |
| `render_oar_curate_dict.py` | `vc_genome/scripts/` | Renders extracted JSON to SVG and PNG scene graphs |
| `export_curate_dict_csv.py` | `vc_genome/scripts/` | Joins extraction JSON with data CSV and exports to spreadsheet-ready CSV |

---

## Output Files

| File | Location | Description |
|---|---|---|
| `oar_curate_dict_9.json` | `vc_genome_output_full/three_conditions/` | Raw LLM extraction for the 9-image sample |
| `curate_dict_9_sample_oar.csv` | `vc_genome/export/` | Spreadsheet export: all input columns + extracted attributes and relationships |
| `*.svg` | `vc_genome/scene_graphs/svg/curated_dict/` | Scene graph SVGs, one per image, named by original image filename |
| `*.png` | `vc_genome/scene_graphs/png/curated_dict/` | Scene graph PNGs, one per image, named by original image filename |

**Scene graph PNG URL prefix (GitHub):**
```
https://raw.githubusercontent.com/mling216/comment-post-processing/main/vc_genome/scene_graphs/png/curated_dict/
```
