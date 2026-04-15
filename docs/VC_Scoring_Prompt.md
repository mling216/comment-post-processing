# Visual Complexity (VC) Scoring Prompt

## Zero-Shot Prompt

Copy everything below the line into a chat window for VC scoring tasks.

---

You are a **Visual Complexity (VC) scoring expert**. Given a visualization image, you will assess its perceived visual complexity on a scale from **0.00** (simplest) to **1.00** (most complex).

### Scoring Framework

Rate the image across **7 dimensions**, each scored 0.0–1.0. The final VC score is their weighted average.

| # | Dimension | Weight | What to evaluate | Low (→ 0) | High (→ 1) |
|---|-----------|--------|-------------------|-----------|-------------|
| 1 | **Data Density** | 0.20 | How much data is packed into the image? Consider: information volume (sparse vs. dense/layered data), element quantity (few vs. many discrete marks — points, lines, bars, shapes, subplots), and visual clutter (overlapping elements, layout congestion, lack of whitespace). | Few data points, ample whitespace, no overlap | Dense data, many overlapping elements, congested layout |
| 2 | **Visual Encoding** | 0.15 | How varied and complex are the graphical forms? Consider: variety of shapes, lines, and mark types; size/thickness/curvature variation; spatial layout organization; and whether encodings are easily decodable or require effort to interpret. | Simple, uniform marks (e.g., plain bars), well-organized layout | Many different mark types with varying sizes, irregular shapes, disorganized positioning |
| 3 | **Text & Annotation** | 0.15 | How much text is present and how legible is it? Consider: presence/absence of labels, axes, legends, captions; the sheer volume of text and numbers; and typography clarity (font size, rotation, density of text). | Clear labels, minimal text, readable fonts | Dense annotations, many small writings, rotated or overlapping text, missing labels that increase ambiguity |
| 4 | **Domain & Schema** | 0.15 | How much specialized knowledge is needed? Consider: domain familiarity (common chart vs. specialized scientific visualization), dimensionality (2D vs. 3D), abstraction level, and whether the representational conventions are familiar or unusual. | Familiar chart type (bar, line, pie), common domain | Specialized/unfamiliar domain, 3D structure, abstract or non-standard representation |
| 5 | **Color & Symbol** | 0.15 | How complex is the color and symbol usage? Consider: number and variety of colors, color contrast and distinguishability, use of gradients/hues/saturation, background effects, and presence of non-color markers (textures, icons, patterns). | 1–3 clearly distinguishable colors, simple fills, no textures | Many colors (especially similar/hard to distinguish), gradients, layered/overlapping color regions, additional symbol/texture encoding |
| 6 | **Aesthetic Order** | 0.10 | Does the image feel organized or chaotic? Consider: visual disorganization (random, messy, overwhelming feeling), coherent structure, and perceptual ambiguity (uncertainty in perceiving visual attributes). | Clean, structured, predictable layout | Messy, distracting, inconsistent, random-feeling arrangement |
| 7 | **Cognitive Load** | 0.10 | How much mental effort does interpretation require? Consider: overall interpretive difficulty, semantic clarity (is the message obvious or confusing?), and processing time needed (quick glance vs. extended study). | Instantly understandable at a glance | Requires extended study, unclear meaning, hard to differentiate or compare elements |

### Scoring Procedure

1. **Examine the image** carefully for 10–15 seconds as a first impression.
2. **Rate each of the 7 dimensions** on a 0.0–1.0 scale (to one decimal place).
3. **Compute the weighted VC score**:
   - `VC = 0.20×DataDensity + 0.15×Encoding + 0.15×Text + 0.15×Domain + 0.15×Color + 0.10×Aesthetic + 0.10×Cognitive`
4. **Report** in this exact format:

```
## VC Assessment: [image name]

| Dimension | Score | Key Observations |
|-----------|-------|-------------------|
| Data Density (0.20) | X.X | ... |
| Visual Encoding (0.15) | X.X | ... |
| Text & Annotation (0.15) | X.X | ... |
| Domain & Schema (0.15) | X.X | ... |
| Color & Symbol (0.15) | X.X | ... |
| Aesthetic Order (0.10) | X.X | ... |
| Cognitive Load (0.10) | X.X | ... |

**VC Score: X.XX**

Brief justification (2–3 sentences).
```

### Calibration Anchors

Use these anchors to calibrate your scores:

- **VC ≈ 0.20**: A simple bar chart with 3–5 bars, clear axis labels, 2–3 colors, instantly readable. (e.g., a basic horizontal bar chart comparing 4 categories)
- **VC ≈ 0.40**: A line chart with moderate detail — multiple lines, some annotations, familiar domain, a few colors, takes ~5 seconds to understand the main message.
- **VC ≈ 0.60**: A multi-panel or multi-encoding chart with varied shapes, several colors that may be somewhat similar, some clutter, requires 10–20 seconds to interpret.
- **VC ≈ 0.80**: A dense node-link diagram or hierarchical treemap with many overlapping elements, specialized domain, many colors hard to distinguish, requires careful study.
- **VC ≈ 0.95**: A maximum-density visualization with overwhelming textual and graphical information — nothing is immediately clear, multiple encoding types, dense overlapping elements across the entire image, requires extended study.

Now score the visualization image I provide.

---

## Few-Shot Calibration Workflow

Use this 2-step procedure to anchor the LLM before scoring new images. You need **two reference images** with known ground-truth VC scores — one low-complexity and one high-complexity.

**Recommended references** (from the BeauVis study):
- Low anchor: `VisC.503.6.png` — simple bar chart, VC = 0.22
- High anchor: `InfoVisJ.1149.6(1).png` — dense grid visualization, VC = 0.95

---

### Step 1 — Blind scoring (one message per reference)

Send the **zero-shot prompt above** together with the first reference image. Let the LLM produce its own score and dimension breakdown. Then do the same for the second reference image.

> **Message 1:** [paste zero-shot prompt] + [attach low-anchor image]
>
> *(LLM responds with its own VC score, e.g. 0.28)*
>
> **Message 2:** [attach high-anchor image] "Now score this image using the same framework."
>
> *(LLM responds with its own VC score, e.g. 0.87)*

### Step 2 — Ground-truth correction (single message)

After both blind scores are returned, tell the LLM the actual scores and ask it to internalize the offset:

> **Message 3:**
>
> Thank you. Here are the ground-truth VC scores for those two images, derived from human evaluations:
>
> | Image | Your Score | Ground Truth |
> |-------|-----------|--------------|
> | VisC.503.6.png | [LLM's score] | **0.22** |
> | InfoVisJ.1149.6(1).png | [LLM's score] | **0.95** |
>
> Please recalibrate your internal scale so that these anchors are respected. For all subsequent images:
> - Use these two images as endpoint references.
> - A new image's VC score should fall between them proportionally — closer to the low reference if it shares more traits with it, closer to the high reference if it resembles that instead.
> - Adjust your dimension scoring tendencies accordingly (e.g., if you overestimated the low-anchor, slightly lower your baseline for each dimension).
>
> Confirm you have recalibrated, then I will send the next image.

### Step 3 — Score new images

From this point, simply send each new image:

> **Message 4+:** "Score this visualization." + [attach new image]
>
> The LLM will produce calibrated scores anchored to the two references.

---

### Why this works

- **Step 1** exposes the LLM's raw scoring bias (e.g., tends to cluster around 0.4–0.7).
- **Step 2** gives the LLM explicit correction data — it can see *how far off* it was on two extremes and adjust proportionally.
- **Step 3** benefits from the adjusted internal calibration for the remainder of the session.

This mirrors how human raters are calibrated: score a few "gold standard" items, receive feedback, then proceed to the full set.

---

## Notes

- **Weights** are derived from the frequency of participant mentions across ~520 visualization images in the BeauVis study. Data Density is weighted highest (0.20) because participants most frequently cited information volume and element quantity when judging complexity.
- **Sentiment convention**: Traits that *increase* complexity count as (+), traits that *decrease* complexity count as (−). The VC score reflects the balance — images dominated by (−) traits score low, images dominated by (+) traits score high.
- **The 7 dimensions condense 19 subtopics** from the full taxonomy:
  1. Data Density = Information Volume + Element Quantity + Visual Clutter & Overlap
  2. Visual Encoding = Graphical Forms & Primitives + Position/Scale/Organization + Encoding Interpretability
  3. Text & Annotation = Annotations & Labels + Text Volume & Content + Typography & Readability
  4. Domain & Schema = Domain Familiarity + Dimensionality & Structure + Abstraction Level
  5. Color & Symbol = Color Palette & Contrast + Symbols & Texture
  6. Aesthetic Order = Visual Disorganization + Perceptual Ambiguity
  7. Cognitive Load = Interpretive Difficulty + Semantic Clarity + Processing Time & Effort
