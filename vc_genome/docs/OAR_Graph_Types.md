# OAR Graph Types

This document maps the rendered OAR (Object-Attribute-Relationship) scene-graph folders under `vc_genome/scene_graphs/png` back to their extraction conditions and rendering scripts.

## Folder Summary

| Folder | Condition | Input to the LLM | Main constraints | Scope |
|---|---|---|---|---|
| `B` | Human-phrase baseline | Participant complexity phrases with sentiment markers; no image | Seven fixed complexity topics; objects and predicates are otherwise model-generated | Three-condition OAR baseline |
| `V1` | Vision-only | Visualization image; no participant phrases | Seven fixed complexity topics; extraction must be grounded in visible content | Three-condition vision comparison |
| `curated_dict` | Curated-dictionary phrase condition | Participant phrases, preferred object words, and image-specific subtopics | Preferred object vocabulary and per-image subtopics | Nine-image pilot |
| `pure_genome` | Full pure-genome phrase condition | Participant phrases from `CuratePhrasesMore/LessComplex` | Shared vocabulary dictionary built from the full corpus; no per-image topic context | Full corpus, 497 rendered images |
| `topic_genome` | Topic-genome phrase condition | The same participant phrases and shared vocabulary as `pure_genome`, plus each image's `UniqueTopics` | Topics provide context but cannot justify concepts absent from the phrases | Nine-image pilot, plus a later single-image probe |
| `pure_genome_0` | Legacy pure-genome pilot output | Same pure-genome extraction as the pilot | Same as `pure_genome`; the suffix is historical, not methodological | Original nine-image pilot, plus one later probe render |

The folder file counts are render artifacts, not necessarily the size of the underlying extraction datasets. For example, `B` is rendered only for a subset of the 510-image JSON, while `pure_genome` contains the full 497-image rendering.

## Detailed Conditions

### `B`

`B` is the human-phrase baseline in the three-condition experiment. The model receives the original complexity phrases extracted from participant comments, paired with `(+/-)` sentiment markers. It does not receive the visualization image.

Attributes and relationships are tagged with one of seven topics:

- Data Density / Image Clutter
- Visual Encoding Clarity
- Semantics / Text Legibility
- Schema
- Color, Symbol, and Texture Details
- Aesthetics Uncertainty
- Immediacy / Cognitive Load

The source extraction is `vc_genome_output_full/three_conditions/oar_B_510.json`. The condition definition is documented in [`paper/OAR_Extraction_Prompts.md`](../../paper/OAR_Extraction_Prompts.md#condition-b-human-phrases-only-baseline), and the renderer is [`code/render_oar_B.py`](../code/render_oar_B.py).

### `V1`

`V1` is the vision-only condition. The model receives the rendered visualization image and must extract the visible objects, attributes, and relationships without participant phrases. It uses the same seven-topic taxonomy as `B`.

The source extraction is `vc_genome_output_full/three_conditions/oar_V1_510.json`. The condition definition is documented in [`paper/OAR_Extraction_Prompts.md`](../../paper/OAR_Extraction_Prompts.md#condition-v1-vision-only). The pilot renderer is [`scripts/render_oar_B_V1.py`](../scripts/render_oar_B_V1.py), which renders the `V1` subset into the `V1` folder.

### `curated_dict`

`curated_dict` is a more tightly grounded phrase condition. For each image, the prompt includes:

- Participant complexity phrases.
- Preferred object words derived from participant descriptions of that image.
- Only the subtopics associated with that image.

Object names are soft-closed: the model is instructed to prefer the supplied vocabulary but may fall back to a specific snake_case name. Subtopics are closed to the image-specific list. Attributes and relationship predicates remain free-form snake_case text. Sentiment is not included in this condition.

The source extraction is `vc_genome_output_full/three_conditions/oar_curate_dict_9.json`. The method is documented in [`OAR_CurateDict_Method.md`](OAR_CurateDict_Method.md), and the renderer is [`scripts/render_oar_curate_dict.py`](../scripts/render_oar_curate_dict.py).

### `pure_genome`

`pure_genome` uses participant phrases, but grounds the extraction with a vocabulary shared across the corpus rather than vocabulary selected separately for each image.

The shared vocabulary is built in two steps:

1. [`code/build_genome_vocab.py`](../code/build_genome_vocab.py) extracts nouns, verbs, adjectives, and noun phrases from the full set of participant phrases.
2. [`code/build_genome_dict.py`](../code/build_genome_dict.py) organizes those terms into a hierarchical dictionary saved as `vc_genome/export/pure_genome_dict.json`.

[`code/_extract_oar_pure_genome.py`](../code/_extract_oar_pure_genome.py) then supplies this dictionary as preferred vocabulary while sending only the participant phrases for the target image. It explicitly does not provide per-image topics or subtopics. The output schema therefore contains objects, attributes, and relationships without topic or subtopic labels.

The pilot output is `vc_genome/export/oar_pure_genome_9.json`; the full output is `vc_genome/export/oar_pure_genome_full.json`. The full rendered graphs are produced by [`code/render_oar_pure_genome.py`](../code/render_oar_pure_genome.py) and stored in `scene_graphs/png/pure_genome`.

### `topic_genome`

`topic_genome` keeps the corpus-wide vocabulary from `pure_genome`, but adds the image's high-level `UniqueTopics` to the prompt. These topics act as a semantic frame for interpreting the phrases. The extraction instructions explicitly say that a listed topic cannot cause the model to invent an object or attribute that the phrases do not describe.

The pilot output is `vc_genome/export/oar_topic_genome_9.json`. A later single-image probe is stored as `vc_genome/export/oar_topic_genome_SciVisJ995.json`. The extraction script is [`code/_extract_oar_topic_genome.py`](../code/_extract_oar_topic_genome.py), and the renderer is [`code/render_oar_topic_genome.py`](../code/render_oar_topic_genome.py).

Unlike `B`, `V1`, and `curated_dict`, the topic-genome renderer displays only the OAR text itself. The topics are prompt context rather than a second line printed inside each attribute or relationship node.

### `pure_genome_0`

`pure_genome_0` is a historical output directory for the nine-image pure-genome pilot. Git history shows the original pilot directory was renamed from `pure_genome_1` to `pure_genome_0`; this rename did not change the extraction condition.

The later full-corpus pure-genome run used the separate `pure_genome` directory. Therefore, `pure_genome_0` should be treated as an earlier pilot render, not as a sixth OAR methodology.

One additional `SciVisJ.995.5` render was added later during a single-image probe commit. It is an extra render artifact and does not define a new condition.

## Relationship Between the Types

The conditions can be grouped by their source modality:

```text
Participant phrases
  B                 phrases + sentiment + seven topics
  curated_dict      phrases + image-specific object vocabulary + image-specific subtopics
  pure_genome       phrases + corpus-wide vocabulary
  topic_genome      phrases + corpus-wide vocabulary + image-level topics

Visualization image
  V1                image + seven topics
```

The six folder names therefore represent four substantive phrase-based variants, one vision-based variant, and one historical output name:

- `B`, `curated_dict`, `pure_genome`, and `topic_genome` are distinct phrase-grounding strategies.
- `V1` is the image-only comparison condition.
- `pure_genome_0` is an older pilot rendering of `pure_genome`, not a separate extraction strategy.

## Source Files

- [`paper/OAR_Extraction_Prompts.md`](../../paper/OAR_Extraction_Prompts.md) - definitions of `B`, `V1`, and `V2`.
- [`OAR_CurateDict_Method.md`](OAR_CurateDict_Method.md) - curated-dictionary condition.
- [`code/_extract_oar_pure_genome.py`](../code/_extract_oar_pure_genome.py) - pure-genome extraction.
- [`code/_extract_oar_topic_genome.py`](../code/_extract_oar_topic_genome.py) - topic-genome extraction.
- [`code/render_oar_B.py`](../code/render_oar_B.py) - `B` rendering.
- [`scripts/render_oar_B_V1.py`](../scripts/render_oar_B_V1.py) - `B` and `V1` pilot rendering.
- [`scripts/render_oar_curate_dict.py`](../scripts/render_oar_curate_dict.py) - curated-dictionary rendering.
- [`code/render_oar_pure_genome.py`](../code/render_oar_pure_genome.py) - pure-genome rendering.
- [`code/render_oar_topic_genome.py`](../code/render_oar_topic_genome.py) - topic-genome rendering.
