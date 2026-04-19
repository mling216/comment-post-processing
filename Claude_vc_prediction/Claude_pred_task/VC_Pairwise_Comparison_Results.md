# Visual Complexity (VC) Pairwise Image Comparison Results

**Evaluator:** Claude Opus 4.6  
**Date:** April 2026  
**Method:** Taxonomy-based evaluation using 7 topics / 19 subtopics  
**Image source:** `https://raw.githubusercontent.com/c109363/ExperimentImage/main/AllDataResize/{image_name}`

---

## Evaluation Framework

Each image pair is assessed across a 7-topic, 19-subtopic VC taxonomy. For each subtopic, the image judged more complex on that dimension is recorded as `Image1`, `Image2`, or `Tie`. The overall `MoreComplex` verdict reflects the image that wins on more subtopics, weighted by the severity of differences.

### Taxonomy Reference

| # | Topic | SubTopic | Description |
|---|---|---|---|
| 1.1 | Data Density / Image Clutter | Information Volume | Perceived amount, richness, or depth of data content |
| 1.2 | Data Density / Image Clutter | Element Quantity | Number of discrete graphical elements (points, lines, bars, shapes, subplots) |
| 1.3 | Data Density / Image Clutter | Visual Clutter & Overlap | Spatial density, overlapping elements, layout congestion, whitespace |
| 2.1 | Visual Encoding Clarity | Graphical Forms & Primitives | Variety, type, and physical attributes of shapes, lines, and mark types |
| 2.2 | Visual Encoding Clarity | Position, Scale & Organization | Spatial layout, alignment, ordering, scale consistency |
| 2.3 | Visual Encoding Clarity | Encoding Interpretability | Whether visual encodings convey clear, decodable meaning |
| 3.1 | Semantics / Text Legibility | Annotations & Labels | Presence, clarity, and sufficiency of titles, axis labels, legends, captions |
| 3.2 | Semantics / Text Legibility | Text Volume & Content | Quantity of text, numbers, or contextual descriptions |
| 3.3 | Semantics / Text Legibility | Typography & Readability | Font size, rotation, structure, spatial arrangement of text |
| 4.1 | Schema | Domain Familiarity | Whether the image requires specialized domain knowledge |
| 4.2 | Schema | Dimensionality & Structure | 2D vs 3D, layout organization, structural framework |
| 4.3 | Schema | Abstraction Level | Use of non-representational, abstract visual forms |
| 5.1 | Color, Symbol & Texture | Color Palette & Contrast | Range, variety, distinguishability of colors |
| 5.2 | Color, Symbol & Texture | Symbols & Texture | Non-color graphical markers, textures, icons, patterns |
| 6.1 | Aesthetics Uncertainty | Visual Disorganization | Whether the image feels random, messy, distracting |
| 6.2 | Aesthetics Uncertainty | Perceptual Ambiguity | Uncertainty in perceiving visual attributes like color and contrast |
| 7.1 | Immediacy / Cognitive Load | Interpretive Difficulty | Overall ease or difficulty of interpreting the visualization |
| 7.2 | Immediacy / Cognitive Load | Semantic Clarity | Whether the visualization conveys an unambiguous message |
| 7.3 | Immediacy / Cognitive Load | Processing Time & Effort | Time and cognitive effort needed to process the visualization |

---

## Results Summary

| Pair | Image 1 | Image 2 | More Complex | Brief Rationale |
|---|---|---|---|---|
| 1 | SciVisJ.822.18.png | wsj603.png | **Image 2** | WSJ line chart has more text, color variety, and callout annotations vs. simple heatmap grids |
| 2 | VisJ.1949.18(3).png | InfoVisJ.1558.9.png | **Image 1** | 3D glyph array with no labels/legend on black; high abstraction and 3D dimensionality |
| 3 | VASTJ.2012.9(2).png | SciVisJ.1236.10.png | **Image 1** | ThemeRiver + word clouds + pie charts vs. repetitive small-multiple heatmaps |
| 4 | VisJ.1197.5(3).png | VASTJ.1.7.png | **Image 2** | Network graph with zoom insets edges out 3D block landscape on form variety and color |
| 5 | v483_n7391_8_f5.png | InfoVisJ.1236.13.png | **Image 2** | Netherlands cartogram with 15+ hues and dual representations vs. uniform contour plot |
| 6 | VisJ.1499.8.png | InfoVisJ.1558.11(3).png | **Image 1** | Two 3D heart renderings with wireframe + glyphs dominate across all 19 subtopics |
| 7 | visMost573.png | InfoVisJ.349.1.png | **Image 1** | Jerusalem sound map with overlapping rings, 16-entry legend, and paragraphs of text |
| 8 | v487_n7406_7_f3.png | economist_daily_chart_153.png | **Image 1** | Displacement field with triple encoding (colormap + arrows + contours) vs. clean stacked area |
| 9 | InfoVisJ.619.17.png | wsj135.png | **Image 1** | Cereal-nutrition biplot with gradient color + size + regression vs. grouped bar chart |
| 10 | v488_n7412_9_f1.png | VASTC.13.9.png | **Image 2** | NLP topic-word visualization with 7 word clouds + flow lines + grid matrices vs. glacier map |
| 11 | VASTJ.2908.2.png | InfoVisC.211.18.png | **Image 1** | 17×5 heatmap/table with color encoding vs. minimal monochrome histogram |
| 12 | SciVisJ.2926.1.png | VisC.167.11.png | **Image 1** | 3D volume rendering of galaxy/vortex with 5+ color channels vs. clean stacked bar chart |
| 13 | VisJ.1515.12.png | whoJ44.png | **Image 1** | Abstract concentric ring interference pattern with no labels vs. simple 2-line chart |
| 14 | VisJ.1541.1(2).png | InfoVisC.133.5(3).png | **Image 1** | FEM mesh with thousands of triangular facets vs. parallel sets with ~8 ribbons |
| 15 | vis734.png | whoQ44_4.png | **Image 1** | Dense Olympic infographic (~120 rows × 25 cols) vs. 8-country stacked area chart |
| 16 | SciVisJ.980.12(2).png | economist_daily_chart_165.png | **Image 1** | 3D flow visualization with streamlines + glyphs + isosurfaces vs. horizontal bar chart |
| 17 | InfoVisJ.2699.12.png | InfoVisC.73.6.png | **Image 1** | Node-link network (~300+ nodes, 500+ edges) vs. text rendering demo on curved splines |
| 18 | v488_n7409_19_f1.png | wsj340.png | **Image 1** | Genomic oncoprint (~3500 cells, 10+ color encodings) vs. WSJ pie chart (6 slices) |
| 19 | VASTJ.422.7.png | VisC.503.6.png | **Image 1** | User event timeline (11 users × 5 color-coded types) vs. 3-bar chart with error bars |
| 20 | InfoVisJ.2402.12(1).png | whoO06_2.png | **Image 1** | Dense treemap/pixel visualization (thousands of rects) vs. grouped horizontal bar chart |
| 21 | v488_n7409_12_f4.png | whoQ50_2.png | **Image 1** | Phylogenetic cladogram + fossil photo + geological timescale vs. monochrome bar chart |
| 22 | InfoVisC.65.5(2).png | visMost97.png | **Image 1** | Dense monochrome DAG (~300+ nodes, 500+ edges) vs. world map infographic |
| 23 | InfoVisJ.1149.6(1).png | SciVisJ.1025.11.png | **Image 1** | Multi-panel text collation tool (7 views, hundreds of cells) vs. 16-dot scatter plot |

---

## Detailed Per-Pair Subtopic Scores

### Pair 1: SciVisJ.822.18.png vs wsj603.png — **Image 2 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Image 1 |
| Element Quantity | Image 2 |
| Visual Clutter & Overlap | Image 2 |
| Graphical Forms & Primitives | Image 2 |
| Position, Scale & Organization | Tie |
| Encoding Interpretability | Image 1 |
| Annotations & Labels | Image 2 |
| Text Volume & Content | Image 2 |
| Typography & Readability | Image 2 |
| Domain Familiarity | Image 1 |
| Dimensionality & Structure | Tie |
| Abstraction Level | Image 1 |
| Color Palette & Contrast | Image 2 |
| Symbols & Texture | Tie |
| Visual Disorganization | Image 2 |
| Perceptual Ambiguity | Image 1 |
| Interpretive Difficulty | Image 1 |
| Semantic Clarity | Image 1 |
| Processing Time & Effort | Image 2 |

> Image2 (WSJ line chart) has far more text (title+subtitle paragraph+annotations+source), more color variety (gold/red/blue/green), multiple callout boxes crowding the right side, and 2 high-frequency time series. Image1 (heatmap grids) is structurally simple but more abstract/ambiguous due to missing legend and unlabeled axes.

---

### Pair 2: VisJ.1949.18(3).png vs InfoVisJ.1558.9.png — **Image 1 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Tie |
| Element Quantity | Image 2 |
| Visual Clutter & Overlap | Image 2 |
| Graphical Forms & Primitives | Image 1 |
| Position, Scale & Organization | Tie |
| Encoding Interpretability | Image 1 |
| Annotations & Labels | Image 2 |
| Text Volume & Content | Image 2 |
| Typography & Readability | Image 2 |
| Domain Familiarity | Image 1 |
| Dimensionality & Structure | Image 1 |
| Abstraction Level | Image 1 |
| Color Palette & Contrast | Image 1 |
| Symbols & Texture | Image 1 |
| Visual Disorganization | Image 2 |
| Perceptual Ambiguity | Image 1 |
| Interpretive Difficulty | Image 1 |
| Semantic Clarity | Image 1 |
| Processing Time & Effort | Image 1 |

> Image1 (3D glyph array on black) is more complex: ~60 3D-rendered crescent glyphs with green/orange/red coloring and no labels/legend/axes at all. High abstraction and domain barrier and 3D dimensionality. Image2 (oriented ellipses scatter) has overlap clutter and annotations but provides a color legend and 2D layout.

---

### Pair 3: VASTJ.2012.9(2).png vs SciVisJ.1236.10.png — **Image 1 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Image 1 |
| Element Quantity | Tie |
| Visual Clutter & Overlap | Image 1 |
| Graphical Forms & Primitives | Image 1 |
| Position, Scale & Organization | Image 1 |
| Encoding Interpretability | Image 1 |
| Annotations & Labels | Image 1 |
| Text Volume & Content | Image 1 |
| Typography & Readability | Image 1 |
| Domain Familiarity | Image 2 |
| Dimensionality & Structure | Tie |
| Abstraction Level | Image 2 |
| Color Palette & Contrast | Image 1 |
| Symbols & Texture | Image 1 |
| Visual Disorganization | Image 1 |
| Perceptual Ambiguity | Tie |
| Interpretive Difficulty | Image 1 |
| Semantic Clarity | Image 2 |
| Processing Time & Effort | Image 1 |

> Image1 (ThemeRiver+word clouds+pie charts) combines stream graph + 2 dense word clouds with 100+ words + pie chart icons + color-coded legend + date timeline. Far more heterogeneous forms and text. Image2 (60 small-multiple heatmaps) has high repetitive element count but uniform encoding (purple-green matrices) and minimal text (10 coded labels).

---

### Pair 4: VisJ.1197.5(3).png vs VASTJ.1.7.png — **Image 2 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Image 2 |
| Element Quantity | Tie |
| Visual Clutter & Overlap | Image 1 |
| Graphical Forms & Primitives | Image 2 |
| Position, Scale & Organization | Image 2 |
| Encoding Interpretability | Tie |
| Annotations & Labels | Image 2 |
| Text Volume & Content | Tie |
| Typography & Readability | Tie |
| Domain Familiarity | Image 1 |
| Dimensionality & Structure | Image 1 |
| Abstraction Level | Tie |
| Color Palette & Contrast | Image 2 |
| Symbols & Texture | Image 2 |
| Visual Disorganization | Image 1 |
| Perceptual Ambiguity | Tie |
| Interpretive Difficulty | Image 1 |
| Semantic Clarity | Image 1 |
| Processing Time & Effort | Image 2 |

> Close pair. Image1 (3D block landscape) has extreme density and 3D occlusion filling ~100% canvas in grayscale with no labels. Image2 (network graph with zoom insets) has more encoding heterogeneity: dense hairball clusters + color-graded node chains (red-orange-yellow) + dashed-circle annotations + zoom triangles. Image2 edges out on form variety and color diversity despite Image1's raw density.

---

### Pair 5: v483_n7391_8_f5.png vs InfoVisJ.1236.13.png — **Image 2 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Tie |
| Element Quantity | Image 2 |
| Visual Clutter & Overlap | Image 2 |
| Graphical Forms & Primitives | Image 2 |
| Position, Scale & Organization | Image 1 |
| Encoding Interpretability | Image 1 |
| Annotations & Labels | Image 1 |
| Text Volume & Content | Image 1 |
| Typography & Readability | Image 1 |
| Domain Familiarity | Image 1 |
| Dimensionality & Structure | Tie |
| Abstraction Level | Tie |
| Color Palette & Contrast | Image 2 |
| Symbols & Texture | Tie |
| Visual Disorganization | Image 2 |
| Perceptual Ambiguity | Image 2 |
| Interpretive Difficulty | Image 1 |
| Semantic Clarity | Image 2 |
| Processing Time & Effort | Image 1 |

> Image2 (Netherlands cartogram + grid map) has more color variety (15+ distinct hues for provinces), two side-by-side representations (choropleth + grid cartogram), 20+ numeric labels, and no legend. Image1 (physics contour plot) has domain complexity (4 axis labels with subscripts like k/kF and E/EF), a 12-level color scale, and overlaid contour lines + diagonal lines, but a more uniform spatial structure.

---

### Pair 6: VisJ.1499.8.png vs InfoVisJ.1558.11(3).png — **Image 1 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Image 1 |
| Element Quantity | Image 1 |
| Visual Clutter & Overlap | Image 1 |
| Graphical Forms & Primitives | Image 1 |
| Position, Scale & Organization | Image 1 |
| Encoding Interpretability | Image 1 |
| Annotations & Labels | Image 1 |
| Text Volume & Content | Image 1 |
| Typography & Readability | Image 1 |
| Domain Familiarity | Image 1 |
| Dimensionality & Structure | Image 1 |
| Abstraction Level | Image 1 |
| Color Palette & Contrast | Image 1 |
| Symbols & Texture | Image 1 |
| Visual Disorganization | Image 1 |
| Perceptual Ambiguity | Image 1 |
| Interpretive Difficulty | Image 1 |
| Semantic Clarity | Image 1 |
| Processing Time & Effort | Image 1 |

> Image1 (two 3D heart renderings with color-mapped surfaces + wireframe grids + glyph markers) is far more complex across all dimensions: 3D perspective rendering; orange-red-yellow gradient; wireframe mesh overlay; green diamond glyphs on the right; domain-specific (cardiac); orientation cubes. Image2 (oriented ellipse scatter in 3 colors on white) is simple: ~150 oriented marks in red/green/blue with mostly whitespace and no labels.

---

### Pair 7: visMost573.png vs InfoVisJ.349.1.png — **Image 1 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Image 1 |
| Element Quantity | Image 1 |
| Visual Clutter & Overlap | Image 1 |
| Graphical Forms & Primitives | Image 1 |
| Position, Scale & Organization | Image 2 |
| Encoding Interpretability | Image 1 |
| Annotations & Labels | Image 1 |
| Text Volume & Content | Image 1 |
| Typography & Readability | Image 1 |
| Domain Familiarity | Image 1 |
| Dimensionality & Structure | Tie |
| Abstraction Level | Tie |
| Color Palette & Contrast | Image 1 |
| Symbols & Texture | Image 1 |
| Visual Disorganization | Image 1 |
| Perceptual Ambiguity | Image 1 |
| Interpretive Difficulty | Image 1 |
| Semantic Clarity | Image 1 |
| Processing Time & Effort | Image 1 |

> Image1 (Jerusalem sound map) is far more complex: overlapping translucent concentric rings in 5+ colors over a city map; 16-entry legend with single + combined sound types; paragraphs of descriptive text; street labels; landmark labels; full-page infographic design. Image2 (orthogonal node-link diagram) is structurally intricate but monochrome gray with only circles/squares/lines and zero text labels.

---

### Pair 8: v487_n7406_7_f3.png vs economist_daily_chart_153.png — **Image 1 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Image 1 |
| Element Quantity | Image 1 |
| Visual Clutter & Overlap | Image 1 |
| Graphical Forms & Primitives | Image 1 |
| Position, Scale & Organization | Tie |
| Encoding Interpretability | Image 1 |
| Annotations & Labels | Image 2 |
| Text Volume & Content | Image 2 |
| Typography & Readability | Tie |
| Domain Familiarity | Image 1 |
| Dimensionality & Structure | Tie |
| Abstraction Level | Image 1 |
| Color Palette & Contrast | Image 1 |
| Symbols & Texture | Image 1 |
| Visual Disorganization | Image 1 |
| Perceptual Ambiguity | Image 1 |
| Interpretive Difficulty | Image 1 |
| Semantic Clarity | Image 1 |
| Processing Time & Effort | Image 1 |

> Image1 (granular-mechanics displacement field) triple-encodes data: rainbow colormap for Δz displacement, black arrow glyphs for direction, and contour isolines — all overlaid on ~100% of the canvas with a rod schematic. High domain barrier (soil/physics). Image2 (Economist stacked area of weather disasters) has 3 clean blue-teal series with standard layout, clear title/legend/source. Image2 has more text/annotations but Image1 dominates on density, form variety, color range, and interpretive load.

---

### Pair 9: InfoVisJ.619.17.png vs wsj135.png — **Image 1 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Image 1 |
| Element Quantity | Image 1 |
| Visual Clutter & Overlap | Image 1 |
| Graphical Forms & Primitives | Image 1 |
| Position, Scale & Organization | Image 1 |
| Encoding Interpretability | Image 1 |
| Annotations & Labels | Image 2 |
| Text Volume & Content | Image 2 |
| Typography & Readability | Image 2 |
| Domain Familiarity | Image 1 |
| Dimensionality & Structure | Image 1 |
| Abstraction Level | Image 1 |
| Color Palette & Contrast | Image 1 |
| Symbols & Texture | Image 1 |
| Visual Disorganization | Image 1 |
| Perceptual Ambiguity | Image 1 |
| Interpretive Difficulty | Image 1 |
| Semantic Clarity | Image 1 |
| Processing Time & Effort | Image 1 |

> Image1 (cereal-nutrition biplot) encodes ~70 dots via position + continuous black-to-orange gradient color + variable size, overlaid with 4 gray loading arrows (Fat/Sugar/Calories/Sodium), a regression line with 6 numeric tick marks, dotted crosshairs, and 2 outlier labels. Requires statistical literacy (PCA biplot + regression). Image2 (WSJ grouped bar chart) is a clean 2-color (green/orange) bar chart with 12 year pairs, a bold title, 3-line subtitle, and 2 callout annotations. Image2 wins only on text volume/annotations.

---

### Pair 10: v488_n7412_9_f1.png vs VASTC.13.9.png — **Image 2 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Image 2 |
| Element Quantity | Image 2 |
| Visual Clutter & Overlap | Image 2 |
| Graphical Forms & Primitives | Image 2 |
| Position, Scale & Organization | Image 2 |
| Encoding Interpretability | Image 2 |
| Annotations & Labels | Tie |
| Text Volume & Content | Image 2 |
| Typography & Readability | Image 2 |
| Domain Familiarity | Image 2 |
| Dimensionality & Structure | Tie |
| Abstraction Level | Image 2 |
| Color Palette & Contrast | Image 2 |
| Symbols & Texture | Image 2 |
| Visual Disorganization | Image 2 |
| Perceptual Ambiguity | Image 2 |
| Interpretive Difficulty | Image 2 |
| Semantic Clarity | Image 2 |
| Processing Time & Effort | Image 2 |

> Image2 (NLP topic-word visualization) is far more complex: 7 word clouds (100+ words total) at varying font sizes/colors mapped to 10 POS categories (VERB/NOUN/ADJ/etc.), connected by ~30+ colored flow lines (blue/red/orange) to small colored grid matrices on the left, plus 4 zoomed-in callout boxes on the right. Multiple representation types (grid glyphs + word clouds + flow ribbons + zoom panels). Image1 (Himalaya glacier map) has ~50 bivariate glyphs on a shaded-relief base map but uses a single consistent encoding.

---

### Pair 11: VASTJ.2908.2.png vs InfoVisC.211.18.png — **Image 1 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Image 1 |
| Element Quantity | Image 1 |
| Visual Clutter & Overlap | Image 1 |
| Graphical Forms & Primitives | Image 1 |
| Position, Scale & Organization | Tie |
| Encoding Interpretability | Image 1 |
| Annotations & Labels | Image 1 |
| Text Volume & Content | Image 1 |
| Typography & Readability | Image 1 |
| Domain Familiarity | Image 1 |
| Dimensionality & Structure | Tie |
| Abstraction Level | Tie |
| Color Palette & Contrast | Image 1 |
| Symbols & Texture | Tie |
| Visual Disorganization | Tie |
| Perceptual Ambiguity | Image 1 |
| Interpretive Difficulty | Image 1 |
| Semantic Clarity | Image 1 |
| Processing Time & Effort | Image 1 |

> Image1 (research-focus heatmap/table) is a 17-row × 5-column matrix where each cell uses 3 shades of blue (high/medium/low) plus gray (absent). 17 long row labels and 5 column headers, plus a legend. ~85 data cells to decode. Image2 (histogram of thread sizes) is a minimal monochrome bar chart: ~18 black bars on white with only 2 axis labels and blurry tick marks. Extremely sparse.

---

### Pair 12: SciVisJ.2926.1.png vs VisC.167.11.png — **Image 1 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Image 1 |
| Element Quantity | Image 1 |
| Visual Clutter & Overlap | Image 1 |
| Graphical Forms & Primitives | Image 1 |
| Position, Scale & Organization | Image 1 |
| Encoding Interpretability | Image 1 |
| Annotations & Labels | Image 2 |
| Text Volume & Content | Image 2 |
| Typography & Readability | Image 2 |
| Domain Familiarity | Image 1 |
| Dimensionality & Structure | Image 1 |
| Abstraction Level | Image 1 |
| Color Palette & Contrast | Image 1 |
| Symbols & Texture | Image 1 |
| Visual Disorganization | Image 1 |
| Perceptual Ambiguity | Image 1 |
| Interpretive Difficulty | Image 1 |
| Semantic Clarity | Image 1 |
| Processing Time & Effort | Image 1 |

> Image1 (3D volume rendering of galaxy/vortex simulation) fills ~60% of canvas with a swirling 3D structure on black background: blue/red/orange/yellow/green color channels, hundreds of glyph-like spikes, translucent layering, and no legend/axes/labels. Extremely high density, 3D occlusion, and domain-specific (astrophysics/fluid dynamics). Image2 (stacked bar chart of isovalue timing) is a clean 3-color stacked bar chart with 30 bars, standard layout, easy to decode.

---

### Pair 13: VisJ.1515.12.png vs whoJ44.png — **Image 1 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Image 1 |
| Element Quantity | Tie |
| Visual Clutter & Overlap | Image 1 |
| Graphical Forms & Primitives | Image 1 |
| Position, Scale & Organization | Image 1 |
| Encoding Interpretability | Image 1 |
| Annotations & Labels | Image 2 |
| Text Volume & Content | Image 2 |
| Typography & Readability | Image 2 |
| Domain Familiarity | Image 1 |
| Dimensionality & Structure | Tie |
| Abstraction Level | Image 1 |
| Color Palette & Contrast | Image 1 |
| Symbols & Texture | Tie |
| Visual Disorganization | Image 1 |
| Perceptual Ambiguity | Image 1 |
| Interpretive Difficulty | Image 1 |
| Semantic Clarity | Image 1 |
| Processing Time & Effort | Image 1 |

> Image1 (concentric ring interference pattern) is a full-canvas abstract image: concentric dark-green/olive/red/brown rings radiating from center with gradient blending and no labels/axes/legend whatsoever. Semantically opaque. Image2 (WHO ODA line chart) is a simple 2-line chart with a descriptive title, labeled axes, 2-entry legend, and value annotations. Very clear and conventional.

---

### Pair 14: VisJ.1541.1(2).png vs InfoVisC.133.5(3).png — **Image 1 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Image 1 |
| Element Quantity | Image 1 |
| Visual Clutter & Overlap | Image 1 |
| Graphical Forms & Primitives | Image 1 |
| Position, Scale & Organization | Image 1 |
| Encoding Interpretability | Image 1 |
| Annotations & Labels | Tie |
| Text Volume & Content | Tie |
| Typography & Readability | Tie |
| Domain Familiarity | Image 1 |
| Dimensionality & Structure | Tie |
| Abstraction Level | Image 1 |
| Color Palette & Contrast | Image 1 |
| Symbols & Texture | Image 1 |
| Visual Disorganization | Image 1 |
| Perceptual Ambiguity | Image 1 |
| Interpretive Difficulty | Image 1 |
| Semantic Clarity | Image 1 |
| Processing Time & Effort | Image 1 |

> Image1 (FEM triangular mesh simulation) fills ~90% of the canvas with thousands of irregular triangular facets color-mapped via a white-to-dark-blue sequential scale (0–18). Extreme element count, high spatial density, and a T-shaped void in the center. Domain-specific (computational fluid dynamics). Image2 (parallel sets / Titanic mosaic plot) shows 3 categorical variables connected by crossing green/teal ribbons. ~8 flow ribbons, 6 category labels. Moderate overlap.

---

### Pair 15: vis734.png vs whoQ44_4.png — **Image 1 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Image 1 |
| Element Quantity | Image 1 |
| Visual Clutter & Overlap | Image 1 |
| Graphical Forms & Primitives | Image 1 |
| Position, Scale & Organization | Image 1 |
| Encoding Interpretability | Image 1 |
| Annotations & Labels | Image 1 |
| Text Volume & Content | Image 1 |
| Typography & Readability | Image 1 |
| Domain Familiarity | Tie |
| Dimensionality & Structure | Tie |
| Abstraction Level | Tie |
| Color Palette & Contrast | Image 1 |
| Symbols & Texture | Tie |
| Visual Disorganization | Image 1 |
| Perceptual Ambiguity | Image 1 |
| Interpretive Difficulty | Image 1 |
| Semantic Clarity | Image 1 |
| Processing Time & Effort | Image 1 |

> Image1 (Olympic proportional-glory infographic) is an extremely dense dark-background matrix: ~120 country rows × ~25 Olympic year columns, each cell a colored bar (pink/magenta/gray/blue/cyan) encoding medal-per-capita performance. 120+ country name labels, ~25 year labels, title, subtitle, and source credits. Thousands of colored cells fill nearly the entire canvas. Image2 (WHO stacked area chart) shows 8 African countries over 2000–2011 in a monochromatic brown palette.

---

### Pair 16: SciVisJ.980.12(2).png vs economist_daily_chart_165.png — **Image 1 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Image 1 |
| Element Quantity | Image 1 |
| Visual Clutter & Overlap | Image 1 |
| Graphical Forms & Primitives | Image 1 |
| Position, Scale & Organization | Image 1 |
| Encoding Interpretability | Image 1 |
| Annotations & Labels | Image 2 |
| Text Volume & Content | Image 2 |
| Typography & Readability | Image 2 |
| Domain Familiarity | Image 1 |
| Dimensionality & Structure | Image 1 |
| Abstraction Level | Image 1 |
| Color Palette & Contrast | Image 1 |
| Symbols & Texture | Image 1 |
| Visual Disorganization | Image 1 |
| Perceptual Ambiguity | Image 1 |
| Interpretive Difficulty | Image 1 |
| Semantic Clarity | Image 1 |
| Processing Time & Effort | Image 1 |

> Image1 (3D flow visualization in a box) shows 4 vortex cores inside a transparent 3D bounding box with gray streamlines, ~60+ colored arrow/diamond glyphs, and 3D ellipsoidal isosurfaces. Multiple overlapping encoding layers. 3D perspective with occlusion. No text labels, no legend, no axes — fully domain-specific (computational fluid dynamics). Image2 (Economist horizontal bar chart of organ donor rates) is a clean 2-color stacked horizontal bar chart. Conventional and immediately readable.

---

### Pair 17: InfoVisJ.2699.12.png vs InfoVisC.73.6.png — **Image 1 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Image 1 |
| Element Quantity | Image 1 |
| Visual Clutter & Overlap | Image 1 |
| Graphical Forms & Primitives | Image 1 |
| Position, Scale & Organization | Image 1 |
| Encoding Interpretability | Image 1 |
| Annotations & Labels | Image 2 |
| Text Volume & Content | Image 2 |
| Typography & Readability | Image 2 |
| Domain Familiarity | Image 1 |
| Dimensionality & Structure | Tie |
| Abstraction Level | Image 1 |
| Color Palette & Contrast | Image 1 |
| Symbols & Texture | Tie |
| Visual Disorganization | Image 1 |
| Perceptual Ambiguity | Image 1 |
| Interpretive Difficulty | Image 1 |
| Semantic Clarity | Image 1 |
| Processing Time & Effort | Image 1 |

> Image1 (node-link network graph) has ~300+ purple/pink nodes connected by ~500+ black edges forming many subgraph clusters, with 4 red-circled annotation labels (SG1–SG4). Dense central region with heavy edge overlap. Requires graph/network literacy. Image2 (text rendering demo) shows "IEEE Information Visualization 2005" repeated at decreasing font sizes along curved splines with ~10 red control-point dots. Mostly whitespace, ~4 curve segments.

---

### Pair 18: v488_n7409_19_f1.png vs wsj340.png — **Image 1 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Image 1 |
| Element Quantity | Image 1 |
| Visual Clutter & Overlap | Image 1 |
| Graphical Forms & Primitives | Image 1 |
| Position, Scale & Organization | Image 1 |
| Encoding Interpretability | Image 1 |
| Annotations & Labels | Image 1 |
| Text Volume & Content | Image 1 |
| Typography & Readability | Image 1 |
| Domain Familiarity | Image 1 |
| Dimensionality & Structure | Tie |
| Abstraction Level | Image 1 |
| Color Palette & Contrast | Image 1 |
| Symbols & Texture | Image 1 |
| Visual Disorganization | Image 1 |
| Perceptual Ambiguity | Image 1 |
| Interpretive Difficulty | Image 1 |
| Semantic Clarity | Image 1 |
| Processing Time & Effort | Image 1 |

> Image1 (genomic oncoprint) is a ~100-patient × ~35-row matrix encoding clinical variables (Sex/Age/Histology), medulloblastoma subtypes (WNT/SHH/Group3/Group4), ~20 copy-number alterations (red=gains, blue=losses), and ~15 somatic mutations (orange/black/purple/green) in gene-pathway groups. ~3500+ data cells, 10+ distinct color encodings, multi-section layout, ~100 rotated column IDs, multi-line bottom legend with ~15 entries. Extremely high domain barrier (cancer genomics). Image2 (WSJ pie/donut chart) shows India's electrical power generation by 6 sources. ~5 colors, ~15 text items. Image1 dominates on every dimension.

---

### Pair 19: VASTJ.422.7.png vs VisC.503.6.png — **Image 1 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Image 1 |
| Element Quantity | Image 1 |
| Visual Clutter & Overlap | Image 1 |
| Graphical Forms & Primitives | Image 1 |
| Position, Scale & Organization | Image 1 |
| Encoding Interpretability | Image 1 |
| Annotations & Labels | Image 1 |
| Text Volume & Content | Image 1 |
| Typography & Readability | Tie |
| Domain Familiarity | Image 1 |
| Dimensionality & Structure | Tie |
| Abstraction Level | Image 1 |
| Color Palette & Contrast | Image 1 |
| Symbols & Texture | Image 1 |
| Visual Disorganization | Image 1 |
| Perceptual Ambiguity | Image 1 |
| Interpretive Difficulty | Image 1 |
| Semantic Clarity | Image 1 |
| Processing Time & Effort | Image 1 |

> Image1 (user event timeline) shows 11 users (User 1–11) on y-axis over 0–70 minutes on x-axis, with ~70+ event markers encoded by color (blue/orange/red/black/yellow) and shape (circles/triangles). 7-entry legend, 11 row labels, time axis with gridlines. Image2 (bar chart of surface orientation task accuracy) has only 3 black-hatched bars with red error bars, title, 2 axis labels, 3 x-tick labels. Extremely sparse.

---

### Pair 20: InfoVisJ.2402.12(1).png vs whoO06_2.png — **Image 1 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Image 1 |
| Element Quantity | Image 1 |
| Visual Clutter & Overlap | Image 1 |
| Graphical Forms & Primitives | Image 1 |
| Position, Scale & Organization | Image 1 |
| Encoding Interpretability | Image 1 |
| Annotations & Labels | Image 2 |
| Text Volume & Content | Image 2 |
| Typography & Readability | Image 2 |
| Domain Familiarity | Image 1 |
| Dimensionality & Structure | Tie |
| Abstraction Level | Image 1 |
| Color Palette & Contrast | Image 1 |
| Symbols & Texture | Tie |
| Visual Disorganization | Image 1 |
| Perceptual Ambiguity | Image 1 |
| Interpretive Difficulty | Image 1 |
| Semantic Clarity | Image 1 |
| Processing Time & Effort | Image 1 |

> Image1 (hierarchical treemap/pixel visualization) fills 100% of the canvas with thousands of small rectangles in a nested layout, colored predominantly gray with green and cyan highlights. No visible text labels, no legend, no axes — purely dense visual structure. Extremely high abstraction and perceptual ambiguity. Image2 (WHO horizontal stacked bar chart) shows 19 health risk factors × 3 income levels. Clear title, 19 row labels, 3-entry legend. Standard and readable. Image2 wins on text/annotation clarity.

---

### Pair 21: v488_n7409_12_f4.png vs whoQ50_2.png — **Image 1 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Image 1 |
| Element Quantity | Image 1 |
| Visual Clutter & Overlap | Image 1 |
| Graphical Forms & Primitives | Image 1 |
| Position, Scale & Organization | Image 1 |
| Encoding Interpretability | Image 1 |
| Annotations & Labels | Image 1 |
| Text Volume & Content | Image 1 |
| Typography & Readability | Image 1 |
| Domain Familiarity | Image 1 |
| Dimensionality & Structure | Tie |
| Abstraction Level | Image 1 |
| Color Palette & Contrast | Image 1 |
| Symbols & Texture | Image 1 |
| Visual Disorganization | Image 1 |
| Perceptual Ambiguity | Image 1 |
| Interpretive Difficulty | Image 1 |
| Semantic Clarity | Image 1 |
| Processing Time & Effort | Image 1 |

> Image1 (phylogenetic cladogram) shows 8 insect taxa as a blue branching tree over geological time (400–300 Myr). Features: dashed lines for uncertain relationships, 3 numbered ancestral nodes, 2 shaded gap regions ("Romer's gap"/"Hexapoda gap"), an embedded fossil photograph of Strudiella with red scale bar, and a 3-tier geological timescale at the bottom with ~20 rotated stage names. ~50+ text items total. Image2 (WHO bar chart) shows percentage change in malaria case incidence for ~20 Latin American countries in a single brown color, sorted by value.

---

### Pair 22: InfoVisC.65.5(2).png vs visMost97.png — **Image 1 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Image 1 |
| Element Quantity | Image 1 |
| Visual Clutter & Overlap | Image 1 |
| Graphical Forms & Primitives | Image 1 |
| Position, Scale & Organization | Image 1 |
| Encoding Interpretability | Image 1 |
| Annotations & Labels | Image 2 |
| Text Volume & Content | Image 2 |
| Typography & Readability | Image 2 |
| Domain Familiarity | Image 1 |
| Dimensionality & Structure | Tie |
| Abstraction Level | Image 1 |
| Color Palette & Contrast | Image 2 |
| Symbols & Texture | Tie |
| Visual Disorganization | Image 1 |
| Perceptual Ambiguity | Image 1 |
| Interpretive Difficulty | Image 1 |
| Semantic Clarity | Image 1 |
| Processing Time & Effort | Image 1 |

> Image1 (hierarchical DAG/network) is a dense monochrome directed graph with ~300+ small black dot nodes connected by ~500+ edges (straight + curved arcs). Multiple fan-out tree sub-structures connected by long arching cross-edges. Heavy edge crossing in the center-left. No text labels, no color, no legend — purely topological. Image2 (infographic "If the world were a village of 100 people — NATIONALITY") shows a simple world map with 6 color-coded continents and 5 large numbers + region labels. Poster-like layout with generous whitespace.

---

### Pair 23: InfoVisJ.1149.6(1).png vs SciVisJ.1025.11.png — **Image 1 is more complex**

| SubTopic | Winner |
|---|---|
| Information Volume | Image 1 |
| Element Quantity | Image 1 |
| Visual Clutter & Overlap | Image 1 |
| Graphical Forms & Primitives | Image 1 |
| Position, Scale & Organization | Image 1 |
| Encoding Interpretability | Image 1 |
| Annotations & Labels | Image 1 |
| Text Volume & Content | Image 1 |
| Typography & Readability | Image 1 |
| Domain Familiarity | Image 1 |
| Dimensionality & Structure | Tie |
| Abstraction Level | Image 1 |
| Color Palette & Contrast | Image 1 |
| Symbols & Texture | Tie |
| Visual Disorganization | Image 1 |
| Perceptual Ambiguity | Image 1 |
| Interpretive Difficulty | Image 1 |
| Semantic Clarity | Image 1 |
| Processing Time & Effort | Image 1 |

> Image1 (multi-panel text-variant collation tool) contains 7 coordinated views labeled A–G: (A) BASE TEXT panel with full Latin text and highlighted variants, (B) OVERVIEW dense color-coded matrix of ~22 witnesses × many text positions, (C–D) navigation indicators, (E) PAGE VIEW matrix, (F) LINE VIEW, (G) WORD VIEW. Hundreds of colored cells across multiple encodings, section headers, panel labels, Latin text. Requires textual-criticism/philology domain knowledge. Image2 (scatter plot of rendering benchmarks) has 16 labeled dots in 4 colors on a clean x/y grid (ms vs megabytes). Spacious whitespace. Image1 dominates on every dimension.

---

## Aggregate Statistics

- **Image 1 judged more complex:** 19 / 23 pairs (82.6%)
- **Image 2 judged more complex:** 4 / 23 pairs (17.4%)
- **Unanimous (all 19 subtopics):** Pairs 6, 10, 18 (one side wins every subtopic)
- **Close calls:** Pair 4 (Image2 edges out despite Image1's raw density)
