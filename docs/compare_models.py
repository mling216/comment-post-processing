"""Compare Claude and GPT predictions against ground truth."""
import re
from collections import Counter

# Ground truth from image_pairs_by_vc_diff.xlsx
gt_data = [
    (1, 0.57, 0.57), (2, 0.57, 0.54), (3, 0.67, 0.72), (4, 0.76, 0.83),
    (5, 0.62, 0.53), (6, 0.69, 0.57), (7, 0.77, 0.63), (8, 0.70, 0.53),
    (9, 0.54, 0.35), (10, 0.67, 0.88), (11, 0.52, 0.30), (12, 0.74, 0.50),
    (13, 0.58, 0.31), (14, 0.77, 0.48), (15, 0.84, 0.52), (16, 0.77, 0.43),
    (17, 0.73, 0.36), (18, 0.75, 0.36), (19, 0.78, 0.36), (20, 0.88, 0.45),
    (21, 0.73, 0.28), (22, 0.92, 0.38), (23, 0.95, 0.36),
]

gt = {}
for pair, vc1, vc2 in gt_data:
    if vc1 > vc2: gt[pair] = "Image1"
    elif vc2 > vc1: gt[pair] = "Image2"
    else: gt[pair] = "Tie"

claude = {
    1:"Image2", 2:"Image1", 3:"Image1", 4:"Image2", 5:"Image2",
    6:"Image1", 7:"Image1", 8:"Image1", 9:"Image1", 10:"Image2",
    11:"Image1", 12:"Image1", 13:"Image1", 14:"Image1", 15:"Image1",
    16:"Image1", 17:"Image1", 18:"Image1", 19:"Image1", 20:"Image1",
    21:"Image1", 22:"Image1", 23:"Image1"
}
gpt = {
    1:"Image1", 2:"Image2", 3:"Image1", 4:"Image2", 5:"Image2",
    6:"Image1", 7:"Image1", 8:"Image1", 9:"Image1", 10:"Image2",
    11:"Image1", 12:"Image1", 13:"Image1", 14:"Image2", 15:"Image1",
    16:"Image1", 17:"Image1", 18:"Image1", 19:"Image1", 20:"Image1",
    21:"Image1", 22:"Image1", 23:"Image1"
}

# Claude subtopic scores per pair (from CSV)
claude_subtopics = {
    1: ["Image1","Image2","Image2","Image2","Tie","Image1","Image2","Image2","Image2","Image1","Tie","Image1","Image2","Tie","Image2","Image1","Image1","Image1","Image2"],
    2: ["Tie","Image2","Image2","Image1","Tie","Image1","Image2","Image2","Image2","Image1","Image1","Image1","Image1","Image1","Image2","Image1","Image1","Image1","Image1"],
    3: ["Image1","Tie","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image2","Tie","Image2","Image1","Image1","Image1","Tie","Image1","Image2","Image1"],
    4: ["Image2","Tie","Image1","Image2","Image2","Tie","Image2","Tie","Tie","Image1","Image1","Tie","Image2","Image2","Image1","Tie","Image1","Image1","Image2"],
    5: ["Tie","Image2","Image2","Image2","Image1","Image1","Image1","Image1","Image1","Image1","Tie","Tie","Image2","Tie","Image2","Image2","Image1","Image2","Image1"],
    6: ["Image1"]*19,
    7: ["Image1","Image1","Image1","Image1","Image2","Image1","Image1","Image1","Image1","Image1","Tie","Tie","Image1","Image1","Image1","Image1","Image1","Image1","Image1"],
    8: ["Image1","Image1","Image1","Image1","Tie","Image1","Image2","Image2","Tie","Image1","Tie","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1"],
    9: ["Image1","Image1","Image1","Image1","Image1","Image1","Image2","Image2","Image2","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1"],
    10: ["Image2","Image2","Image2","Image2","Image2","Image2","Tie","Image2","Image2","Image2","Tie","Image2","Image2","Image2","Image2","Image2","Image2","Image2","Image2"],
    11: ["Image1","Image1","Image1","Image1","Tie","Image1","Image1","Image1","Image1","Image1","Tie","Tie","Image1","Tie","Tie","Image1","Image1","Image1","Image1"],
    12: ["Image1","Image1","Image1","Image1","Image1","Image1","Image2","Image2","Image2","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1"],
    13: ["Image1","Tie","Image1","Image1","Image1","Image1","Image2","Image2","Image2","Image1","Tie","Image1","Image1","Tie","Image1","Image1","Image1","Image1","Image1"],
    14: ["Image1","Image1","Image1","Image1","Image1","Image1","Tie","Tie","Tie","Image1","Tie","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1"],
    15: ["Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Tie","Tie","Tie","Image1","Tie","Image1","Image1","Image1","Image1","Image1"],
    16: ["Image1","Image1","Image1","Image1","Image1","Image1","Image2","Image2","Image2","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1"],
    17: ["Image1","Image1","Image1","Image1","Image1","Image1","Image2","Image2","Image2","Image1","Tie","Image1","Image1","Tie","Image1","Image1","Image1","Image1","Image1"],
    18: ["Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Tie","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1"],
    19: ["Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Tie","Image1","Tie","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1"],
    20: ["Image1","Image1","Image1","Image1","Image1","Image1","Image2","Image2","Image2","Image1","Tie","Image1","Image1","Tie","Image1","Image1","Image1","Image1","Image1"],
    21: ["Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Tie","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1"],
    22: ["Image1","Image1","Image1","Image1","Image1","Image1","Image2","Image2","Image2","Image1","Tie","Image1","Image2","Tie","Image1","Image1","Image1","Image1","Image1"],
    23: ["Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Image1","Tie","Image1","Image1","Tie","Image1","Image1","Image1","Image1","Image1"],
}

subtopic_names = [
    "Information Volume", "Element Quantity", "Visual Clutter & Overlap",
    "Graphical Forms & Primitives", "Position, Scale & Organization", "Encoding Interpretability",
    "Annotations & Labels", "Text Volume & Content", "Typography & Readability",
    "Domain Familiarity", "Dimensionality & Structure", "Abstraction Level",
    "Color Palette & Contrast", "Symbols & Texture",
    "Visual Disorganization", "Perceptual Ambiguity",
    "Interpretive Difficulty", "Semantic Clarity", "Processing Time & Effort",
]

topic_map = {
    0: "Data Density", 1: "Data Density", 2: "Data Density",
    3: "Encoding Clarity", 4: "Encoding Clarity", 5: "Encoding Clarity",
    6: "Semantics/Text", 7: "Semantics/Text", 8: "Semantics/Text",
    9: "Schema", 10: "Schema", 11: "Schema",
    12: "Color/Symbol", 13: "Color/Symbol",
    14: "Aesthetics", 15: "Aesthetics",
    16: "Cognitive Load", 17: "Cognitive Load", 18: "Cognitive Load",
}

# GPT reasoning texts
gpt_reasons = {
    1: "Image 1 is higher on element quantity visual density structural repetition and processing effort because the repeated gridded panels create more discrete marks and denser inspection than the simpler two-line annotated chart.",
    2: "Image 2 is higher on information volume annotations and processing effort because it combines many marks with a color scale inset frames and linking guides whereas Image 1 is visually repetitive but structurally simpler.",
    3: "Image 1 is higher on information volume annotation load text volume color variety and processing effort because it combines a flow view multiple labels legend dates and two word-cloud panels while Image 2 is a more uniform matrix display.",
    4: "Image 2 is higher on element quantity graphical variety color complexity and interpretive effort because it combines dense network clusters connector paths highlights and multiple node groups whereas Image 1 is a single repeated 3D block field with less semantic structure.",
    5: "Image 2 is higher on information volume element quantity annotation load color variety and processing effort because it combines a labeled geographic map with a second tiled view and many discrete colored regions whereas Image 1 is a single scientific contour panel with fewer overall components.",
    6: "Image 1 is higher on graphical variety structural complexity annotation load and processing effort because it combines multiple 3D surface panels grid structures labels and inset cues whereas Image 2 is a single sparse field of colored ellipses with lighter semantic framing and more whitespace.",
    7: "Image 1 is higher on information volume color complexity annotation load overlap and processing effort because it overlays many colored circular fields on a labeled map with explanatory text and a long legend whereas Image 2 is a dense but largely monochrome node-link structure with far less semantic and visual layering.",
    8: "Image 1 is higher on element quantity graphical variety color complexity and interpretive effort because it combines a heatmap contour lines dense vector arrows axes and a color scale within one panel whereas Image 2 is a simpler stacked area chart with fewer encoding layers and more regular structure.",
    9: "Image 1 is higher on abstraction level encoding interpretability burden and processing effort because it uses an abstract multivariate point layout with directional axes and sparse reference structure whereas Image 2 is a straightforward grouped bar chart with familiar marks and clearer semantic framing.",
    10: "Image 2 is higher on information volume element quantity annotation load text content and processing effort because it combines many linked modules colored categories and multiple word panels whereas Image 1 is a single annotated map with repeated symbols and a more uniform spatial structure.",
    11: "Image 1 is higher on information volume text load element quantity and processing effort because it combines many labeled rows repeated column structures and multiple color-coded cells whereas Image 2 is a simple single-series bar chart with familiar marks and minimal semantic layering.",
    12: "Image 1 is higher on graphical complexity color richness abstraction and interpretive effort because it uses a dense volumetric scientific rendering with layered textures and irregular structures whereas Image 2 is a familiar stacked bar chart with explicit axes legend and more regular organization.",
    13: "Image 1 is higher on visual density abstraction texture complexity and interpretive ambiguity because it consists of many tightly packed concentric bands with weak semantic cues whereas Image 2 is a familiar two-line chart with clear axes labels and regular organization despite its heavier text framing.",
    14: "Image 2 is higher on information volume annotation load structural organization and processing effort because it combines multiple labeled categorical blocks with crossing flow bands whereas Image 1 is a dense abstract texture field with high visual density but much less explicit semantic structure.",
    15: "Image 1 is higher on information volume element quantity text load and processing effort because it combines many rows columns colored cells and fine-grained labels in a dense matrix-like poster whereas Image 2 is a more regular stacked area chart with fewer encoding layers and a simpler reading path.",
    16: "Image 1 is higher on graphical variety color complexity abstraction and processing effort because it combines multiple flow structures glyphs and streamlines within a scientific 3D scene whereas Image 2 is a straightforward stacked horizontal bar chart with familiar marks and explicit labels.",
    17: "Image 1 is higher on element quantity visual clutter structural complexity and processing effort because it contains many linked nodes edges and highlighted subgraphs across the full field whereas Image 2 is a sparse text-distortion demonstration with far fewer data-bearing components and lighter semantic load.",
    18: "Image 1 is higher on information volume annotation load element quantity and processing effort because it combines many labeled rows columns subgroup divisions and color-coded mutation marks whereas Image 2 is a simple donut chart with limited categories and a short reading path.",
    19: "Image 1 is higher on element quantity graphical variety annotation burden and processing effort because it combines many user rows repeated timelines multicolored marks and a denser legend structure whereas Image 2 is a small set of hatched bars with error bars and much simpler organization.",
    20: "Image 1 is higher on information volume element quantity visual density and processing effort because it contains a very large treemap-like field of many tiny rectangles with color accents and weak whitespace whereas Image 2 is a conventional stacked horizontal bar chart with clear labels and a simpler reading path.",
    21: "Image 1 is higher on annotation load structural complexity semantic content and processing effort because it combines a timeline tree highlighted temporal bands labels and an inset fossil image whereas Image 2 is a simple single-series bar chart with a straightforward ordering and familiar marks.",
    22: "Image 1 is higher on element quantity visual clutter overlap and processing effort because it contains a dense node-link network with many crossings and tightly packed connections whereas Image 2 is a sparse infographic with a few labeled continent shapes and a short reading path.",
    23: "Image 1 is higher on information volume text load structural organization and processing effort because it combines base text overview and line views with many encoded cells labels and linked sections whereas Image 2 is a simpler labeled scatterplot with fewer marks and a more direct reading path.",
}

# Claude reasoning texts (from the CSV Rationale column)
claude_reasons = {
    1: "Image2 (WSJ line chart) has far more text (title+subtitle paragraph+annotations+source), more color variety (gold/red/blue/green), multiple callout boxes crowding the right side, and 2 high-frequency time series. Image1 (heatmap grids) is structurally simple but more abstract/ambiguous due to missing legend and unlabeled axes.",
    2: "Image1 (3D glyph array on black) is more complex: ~60 3D-rendered crescent glyphs with green/orange/red coloring and no labels/legend/axes at all. High abstraction and domain barrier and 3D dimensionality. Image2 (oriented ellipses scatter) has overlap clutter and annotations but provides a color legend and 2D layout.",
    3: "Image1 (ThemeRiver+word clouds+pie charts) combines stream graph + 2 dense word clouds with 100+ words + pie chart icons + color-coded legend + date timeline. Far more heterogeneous forms and text. Image2 (60 small-multiple heatmaps) has high repetitive element count but uniform encoding (purple-green matrices) and minimal text (10 coded labels).",
    4: "Close pair. Image1 (3D block landscape) has extreme density and 3D occlusion filling ~100% canvas in grayscale with no labels. Image2 (network graph with zoom insets) has more encoding heterogeneity: dense hairball clusters + color-graded node chains (red-orange-yellow) + dashed-circle annotations + zoom triangles. Image2 edges out on form variety and color diversity despite Image1s raw density.",
    5: "Image2 (Netherlands cartogram + grid map) has more color variety (15+ distinct hues for provinces), two side-by-side representations (choropleth + grid cartogram), 20+ numeric labels, and no legend. Image1 (physics contour plot) has domain complexity (4 axis labels with subscripts like k/kF and E/EF), a 12-level color scale, and overlaid contour lines + diagonal lines, but a more uniform spatial structure.",
    6: "Image1 (two 3D heart renderings with color-mapped surfaces + wireframe grids + glyph markers) is far more complex across all dimensions: 3D perspective rendering; orange-red-yellow gradient; wireframe mesh overlay; green diamond glyphs on the right; domain-specific (cardiac); orientation cubes. Image2 (oriented ellipse scatter in 3 colors on white) is simple: ~150 oriented marks in red/green/blue with mostly whitespace and no labels.",
    7: "Image1 (Jerusalem sound map) is far more complex: overlapping translucent concentric rings in 5+ colors over a city map; 16-entry legend with single + combined sound types; paragraphs of descriptive text; street labels; landmark labels; full-page infographic design. Image2 (orthogonal node-link diagram) is structurally intricate but monochrome gray with only circles/squares/lines and zero text labels.",
    8: "Image1 (granular-mechanics displacement field) triple-encodes data: rainbow colormap for displacement, black arrow glyphs for direction, and contour isolines all overlaid on ~100% of the canvas with a rod schematic. High domain barrier (soil/physics). Image2 (Economist stacked area of weather disasters) has 3 clean blue-teal series with standard layout, clear title/legend/source. Image2 has more text/annotations but Image1 dominates on density, form variety, color range, and interpretive load.",
    9: "Image1 (cereal-nutrition biplot) encodes ~70 dots via position + continuous black-to-orange gradient color + variable size, overlaid with 4 gray loading arrows, a regression line with 6 numeric tick marks, dotted crosshairs, and 2 outlier labels. Requires statistical literacy (PCA biplot + regression). Image2 (WSJ grouped bar chart) is a clean 2-color bar chart. Image2 wins only on text volume/annotations.",
    10: "Image2 (NLP topic-word visualization) is far more complex: 7 word clouds (100+ words total) at varying font sizes/colors mapped to 10 POS categories, connected by ~30+ colored flow lines to small colored grid matrices on the left, plus 4 zoomed-in callout boxes on the right. Multiple representation types. Image1 (Himalaya glacier map) has ~50 bivariate glyphs but uses a single consistent encoding.",
    11: "Image1 (research-focus heatmap/table) is a 17x5 matrix where each cell uses 3 shades of blue plus gray. 17 long row labels and 5 column headers, plus a legend. ~85 data cells to decode. Image2 (histogram of thread sizes) is a minimal monochrome bar chart: ~18 black bars on white with only 2 axis labels.",
    12: "Image1 (3D volume rendering of galaxy/vortex simulation) fills ~60% of canvas with a swirling 3D structure on black background: blue/red/orange/yellow/green color channels, hundreds of glyph-like spikes, translucent layering. Extremely high density, 3D occlusion, and domain-specific. Image2 (stacked bar chart of isovalue timing) is a clean 3-color stacked bar chart.",
    13: "Image1 (concentric ring interference pattern) is a full-canvas abstract image: concentric dark-green/olive/red/brown rings with gradient blending and no labels/axes/legend. Semantically opaque. Image2 (WHO ODA line chart) is a simple 2-line chart with labeled axes and legend.",
    14: "Image1 (FEM triangular mesh simulation) fills ~90% of the canvas with thousands of irregular triangular facets color-mapped via a white-to-dark-blue sequential scale. Extreme element count, high spatial density. Domain-specific (computational fluid dynamics). Image2 (parallel sets / Titanic mosaic plot) shows 3 categorical variables with ~8 flow ribbons.",
    15: "Image1 (Olympic proportional-glory infographic) is an extremely dense dark-background matrix: ~120 country rows x ~25 Olympic year columns. Thousands of colored cells fill nearly the entire canvas. Image2 (WHO stacked area chart) shows 8 African countries in a monochromatic brown palette.",
    16: "Image1 (3D flow visualization in a box) shows 4 vortex cores with gray streamlines, ~60+ colored arrow/diamond glyphs, and 3D ellipsoidal isosurfaces. Multiple overlapping encoding layers. 3D perspective with occlusion. No text labels. Image2 (Economist horizontal bar chart of organ donor rates) is clean and conventional.",
    17: "Image1 (node-link network graph) has ~300+ purple/pink nodes connected by ~500+ black edges with 4 red-circled annotation labels. Dense central region with heavy edge overlap. Image2 (text rendering demo) is mostly whitespace with ~4 curve segments.",
    18: "Image1 (genomic oncoprint) is a ~100-patient x ~35-row matrix with ~3500+ data cells, 10+ distinct color encodings. Extremely high domain barrier (cancer genomics). Image2 (WSJ pie/donut chart) has ~5 colors, ~15 text items.",
    19: "Image1 (user event timeline) shows 11 users over 0-70 minutes with ~70+ event markers encoded by color and shape. Image2 (bar chart) has only 3 black-hatched bars. Extremely sparse.",
    20: "Image1 (hierarchical treemap/pixel visualization) fills 100% of the canvas with thousands of small rectangles. No visible text labels. Extremely high abstraction. Image2 (WHO horizontal stacked bar chart) is standard and readable.",
    21: "Image1 (phylogenetic cladogram) shows 8 insect taxa as a blue branching tree with dashed lines, shaded gap regions, embedded fossil photograph, and 3-tier geological timescale. ~50+ text items. Image2 (WHO bar chart) is a single brown color sorted bar chart.",
    22: "Image1 (hierarchical DAG/network) is a dense monochrome directed graph with ~300+ nodes, ~500+ edges. Heavy edge crossing. No text labels. Image2 (infographic) is a simple world map with 6 color-coded continents.",
    23: "Image1 (multi-panel text-variant collation tool) contains 7 coordinated views labeled A-G with hundreds of colored cells across multiple encodings. Requires textual-criticism/philology domain knowledge. Image2 (scatter plot) has 16 labeled dots and spacious whitespace.",
}

# ===== ANALYSIS =====
print("=" * 60)
print("ACCURACY vs GROUND TRUTH (excluding Pair 1 = Tie)")
print("=" * 60)
n = 0
c_correct = 0
g_correct = 0
both_correct = 0
c_only_correct = 0
g_only_correct = 0
both_wrong = 0

for p in range(1, 24):
    if gt[p] == "Tie":
        continue
    n += 1
    c_ok = claude[p] == gt[p]
    g_ok = gpt[p] == gt[p]
    if c_ok: c_correct += 1
    if g_ok: g_correct += 1
    if c_ok and g_ok: both_correct += 1
    elif c_ok and not g_ok: c_only_correct += 1
    elif g_ok and not c_ok: g_only_correct += 1
    else: both_wrong += 1

print(f"Non-tie pairs: {n}")
print(f"Claude Opus 4.6:  {c_correct}/{n} correct = {c_correct/n*100:.1f}%")
print(f"GPT 5.4:          {g_correct}/{n} correct = {g_correct/n*100:.1f}%")
print(f"Both correct:     {both_correct}")
print(f"Claude-only right: {c_only_correct}")
print(f"GPT-only right:    {g_only_correct}")
print(f"Both wrong:        {both_wrong}")

# Inter-model agreement
agree = sum(1 for p in range(1,24) if claude[p] == gpt[p])
print(f"\nInter-model agreement: {agree}/23 = {agree/23*100:.1f}%")

# Disagreements
print("\n" + "=" * 60)
print("DISAGREEMENT PAIRS")
print("=" * 60)
for p in range(1, 24):
    if claude[p] != gpt[p]:
        d = gt_data[p-1]
        print(f"Pair {p}: Claude={claude[p]}, GPT={gpt[p]}, GT={gt[p]}, VC diff={abs(d[1]-d[2]):.2f}")

# Accuracy by difficulty bucket
print("\n" + "=" * 60)
print("ACCURACY BY VC DIFFERENCE MAGNITUDE")
print("=" * 60)
buckets = [
    (0.0, 0.10, "Tiny (0.00-0.09)"),
    (0.10, 0.20, "Small (0.10-0.19)"),
    (0.20, 0.40, "Medium (0.20-0.39)"),
    (0.40, 1.00, "Large (0.40+)"),
]
for lo, hi, label in buckets:
    pairs_in = [p for p in range(1,24) if lo <= abs(gt_data[p-1][1]-gt_data[p-1][2]) < hi and gt[p] != "Tie"]
    if not pairs_in:
        ties = [p for p in range(1,24) if lo <= abs(gt_data[p-1][1]-gt_data[p-1][2]) < hi]
        if ties:
            print(f"{label}: {len(ties)} pair(s) — all ties, excluded")
        continue
    c_acc = sum(1 for p in pairs_in if claude[p] == gt[p])
    g_acc = sum(1 for p in pairs_in if gpt[p] == gt[p])
    print(f"{label}: n={len(pairs_in)}, Claude={c_acc}/{len(pairs_in)} ({c_acc/len(pairs_in)*100:.0f}%), GPT={g_acc}/{len(pairs_in)} ({g_acc/len(pairs_in)*100:.0f}%)")

# Claude subtopic analysis
print("\n" + "=" * 60)
print("CLAUDE SUBTOPIC SCORE DISTRIBUTION (across all 23 pairs)")
print("=" * 60)
for i, name in enumerate(subtopic_names):
    img1 = sum(1 for p in range(1,24) if claude_subtopics[p][i] == "Image1")
    img2 = sum(1 for p in range(1,24) if claude_subtopics[p][i] == "Image2")
    tie = sum(1 for p in range(1,24) if claude_subtopics[p][i] == "Tie")
    print(f"  {name:35s}: Image1={img1:2d}  Image2={img2:2d}  Tie={tie:2d}")

# Topic-level aggregation
print("\n" + "=" * 60)
print("CLAUDE TOPIC-LEVEL IMAGE1 WIN RATE")
print("=" * 60)
topic_names_ordered = ["Data Density", "Encoding Clarity", "Semantics/Text", "Schema", "Color/Symbol", "Aesthetics", "Cognitive Load"]
for tn in topic_names_ordered:
    indices = [i for i, v in topic_map.items() if v == tn]
    total_img1 = 0
    total_img2 = 0
    total_tie = 0
    for p in range(1,24):
        for i in indices:
            v = claude_subtopics[p][i]
            if v == "Image1": total_img1 += 1
            elif v == "Image2": total_img2 += 1
            else: total_tie += 1
    total = total_img1 + total_img2 + total_tie
    print(f"  {tn:20s}: Image1={total_img1:3d} ({total_img1/total*100:4.1f}%)  Image2={total_img2:3d} ({total_img2/total*100:4.1f}%)  Tie={total_tie:3d} ({total_tie/total*100:4.1f}%)")

# Reasoning text analysis
print("\n" + "=" * 60)
print("REASONING TEXT STATISTICS")
print("=" * 60)

def word_count(text):
    return len(text.split())

def sentence_count(text):
    return len(re.split(r'[.!?]+', text.strip())) - (1 if text.strip()[-1] in '.!?' else 0)

c_words = [word_count(claude_reasons[p]) for p in range(1,24)]
g_words = [word_count(gpt_reasons[p]) for p in range(1,24)]
c_sents = [sentence_count(claude_reasons[p]) for p in range(1,24)]
g_sents = [sentence_count(gpt_reasons[p]) for p in range(1,24)]

print(f"Claude avg word count: {sum(c_words)/23:.1f} (min={min(c_words)}, max={max(c_words)})")
print(f"GPT avg word count:    {sum(g_words)/23:.1f} (min={min(g_words)}, max={max(g_words)})")
print(f"Claude avg sentences:  {sum(c_sents)/23:.1f} (min={min(c_sents)}, max={max(c_sents)})")
print(f"GPT avg sentences:     {sum(g_sents)/23:.1f} (min={min(g_sents)}, max={max(g_sents)})")

# Extract subtopic/keyword mentions from GPT reasoning
subtopic_keywords = {
    "information volume": ["information volume", "info volume"],
    "element quantity": ["element quantity", "element count", "discrete marks"],
    "visual clutter": ["visual clutter", "clutter", "overlap", "density", "dense"],
    "graphical forms": ["graphical variety", "graphical forms", "form variety", "mark type"],
    "position/scale": ["position", "scale", "organization", "layout", "spatial"],
    "encoding interpretability": ["encoding interpretability", "encoding", "interpretab"],
    "annotations/labels": ["annotation", "label", "legend", "title", "axis"],
    "text volume": ["text volume", "text load", "text content"],
    "typography": ["typograph", "font", "readab"],
    "domain familiarity": ["domain", "specialized", "scientific"],
    "dimensionality": ["dimension", "2d", "3d", "structure"],
    "abstraction": ["abstract"],
    "color": ["color", "hue", "palette", "contrast"],
    "symbols/texture": ["symbol", "texture", "pattern", "icon", "glyph"],
    "disorganization": ["disorganiz", "messy", "random"],
    "perceptual ambiguity": ["ambiguity", "ambiguous", "uncertain"],
    "interpretive difficulty": ["interpretive", "interpreting", "interpret"],
    "semantic clarity": ["semantic", "meaning", "message"],
    "processing effort": ["processing effort", "processing time", "cognitive"],
}

print("\n" + "=" * 60)
print("SUBTOPIC KEYWORD MENTIONS IN REASONING (approximate)")
print("=" * 60)

for concept, keywords in subtopic_keywords.items():
    c_mentions = 0
    g_mentions = 0
    for p in range(1,24):
        for kw in keywords:
            if kw.lower() in claude_reasons[p].lower():
                c_mentions += 1
                break
        for kw in keywords:
            if kw.lower() in gpt_reasons[p].lower():
                g_mentions += 1
                break
    print(f"  {concept:25s}: Claude={c_mentions:2d}/23  GPT={g_mentions:2d}/23")

# Per-pair detail for the report
print("\n" + "=" * 60)
print("PER-PAIR DETAIL TABLE")
print("=" * 60)
print(f"{'Pair':>4} {'VC1':>5} {'VC2':>5} {'Diff':>5} {'GT':>7} {'Claude':>7} {'GPT':>7} {'C_ok':>5} {'G_ok':>5} {'Agree':>6}")
for p in range(1,24):
    d = gt_data[p-1]
    g = gt[p]
    c = claude[p]
    gp = gpt[p]
    diff = abs(d[1]-d[2])
    c_ok = "Y" if c == g else ("~" if g == "Tie" else "N")
    g_ok = "Y" if gp == g else ("~" if g == "Tie" else "N")
    ag = "Y" if c == gp else "N"
    print(f"{p:4d} {d[1]:5.2f} {d[2]:5.2f} {diff:5.2f} {g:>7} {c:>7} {gp:>7} {c_ok:>5} {g_ok:>5} {ag:>6}")
