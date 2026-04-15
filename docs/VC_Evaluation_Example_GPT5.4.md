# VC Comparison: whoQ58_3 vs. InfoVisJ.619.10

Date: April 7, 2026

## Images

- Image 1: [whoQ58_3.png](https://raw.githubusercontent.com/c109363/ExperimentImage/main/AllDataResize/whoQ58_3.png)
- Image 2: [InfoVisJ.619.10.png](https://raw.githubusercontent.com/c109363/ExperimentImage/main/AllDataResize/InfoVisJ.619.10.png)

## Method 1: Direct Visual Inspection

### Result

Image 1 is more visually complex than Image 2.

Visual complexity (VC) is defined here as "the amount of detail or intricacy."

Image 1 has higher VC because it includes:

- Multiple stacked layers that must be separated visually
- A long legend with several categories
- A time axis and a quantitative y-axis
- Gridlines and labels that add reference structure
- Changing shapes across years that require trend tracking and comparison

This makes the figure denser and more intricate to interpret.

Image 2 is less complex because it has:

- A smaller number of visual components overall
- A few clearly separated point clusters
- Limited color variation
- Fewer annotations and less visual crowding

Its structure is easier to parse at a glance.

### Conclusion

Under the stated definition of VC, Image 1 has higher visual complexity than Image 2.

## Method 2: Dictionary-Based Evaluation

This section uses only the dictionary-derived word stems for the two visualization types and does not rely on the earlier direct visual inspection.

Source dictionary:

- [stem_dictionary_by_vistype_columnar.csv](../phrase_reduction_v2/stem_dictionary_by_vistype_columnar.csv)

For this pair, Image 1 is an Area chart and Image 2 is a Point chart.

The Area-chart dictionary is associated with stems such as shape, detail, layer, density, clutter, overlap, multiple, complicated, decipher, legend, axis, and time. Taken together, these words suggest a chart type that is often experienced as visually denser, more layered, and more demanding to parse.

The Point-chart dictionary also includes some complexity-related stems, such as clutter, overlap, random, and complex, but it more strongly features stems such as clear, easy, simple, intuitive, separate, spread, glance, and discern. These words suggest a chart type that is more readily segmented into individual marks and more easily interpreted at a glance.

Based on this dictionary-only evaluation, Image 1 is more visually complex than Image 2.

Reasoning summary: the Area-chart vocabulary emphasizes layering, density, overlap, and clutter, whereas the Point-chart vocabulary more strongly emphasizes separability, clarity, and quick visual parsing.

## Method 3: Topic-Subtopic Evaluation

This section uses only the 7-topic, 19-subtopic VC framework in the shortlist taxonomy and does not rely on the earlier two methods.

Source taxonomy:

- [phrase_shortlist.csv](../phrase_reduction_v2/phrase_shortlist.csv)

For this pair, Image 1 is evaluated against Image 2 using the shortlist topics and subtopics.

### Result

Image 1 is more visually complex than Image 2.

### Reasoning

Under Data Density / Image Clutter, Image 1 is higher on Information Volume, Element Quantity, and Visual Clutter & Overlap. It contains multiple stacked regions across time, which creates more packed data-bearing structure and more boundaries to parse. Image 2 contains many points, but they are more spatially separated and the overall layout is less crowded.

Under Visual Encoding Clarity, Image 1 is higher on Graphical Forms & Primitives and carries more burden in Position, Scale & Organization. The viewer must track layered filled shapes over time and distinguish component contributions within the stack. Image 2 uses a simpler structure of points and a small number of directional axes, making the encoding easier to segment.

Under Semantics / Text Legibility, Image 1 is higher on Annotations & Labels, Text Volume & Content, and Typography & Readability load. It includes a title, legend, axis labels, tick labels, and numeric scale references. Image 2 has only a few text labels and a lighter semantic frame.

Under Schema, the two images are closer. Both rely on familiar 2D chart conventions, so Domain Familiarity and Dimensionality & Structure do not strongly separate them. Abstraction Level is also not a major differentiator here.

Under Color, Symbol, and Texture Details, Image 1 is higher on Color Palette & Contrast complexity because it uses multiple adjacent filled hues that must be visually distinguished across layers. Image 2 uses fewer colors and clearer cluster separation. Symbols & Texture contributes little additional complexity in either image.

Under Aesthetics Uncertainty, Image 1 is somewhat higher on Visual Disorganization because the stacked layers, multiple labels, and dense colored bands create a busier composition. Image 2 appears more organized, with visible cluster separation and more whitespace. Perceptual Ambiguity is relatively low in both images, though Image 1 presents more opportunities for confusion.

Under Immediacy / Cognitive Load, Image 1 is higher on Interpretive Difficulty, Semantic Clarity burden, and Processing Time & Effort. Understanding it requires integrating categories, time, scale, and part-to-whole relationships. Image 2 can be interpreted more quickly as grouped points arranged along a few labeled directions.

### Conclusion

Using only the shortlist taxonomy, Image 1 has higher VC than Image 2 because it scores higher across more subtopics, especially data density, clutter, annotation load, color-layer complexity, and processing effort.