import json
from collections import Counter

data = json.load(open('vc_genome_output_full/llm_extractions_api.json'))
all_names = [obj['name'] for ext in data.values() for obj in ext.get('objects', [])]
c = Counter(all_names)

MAPPED = set("""bar bars dot point data_point data_points dots shape shapes line lines
vertical_line color_segment glyph glyphs node nodes edge edges area area_marks area_shape
area_chart symbol icon element object color color_shading color_encoding color_pattern
color_palette color_scheme font_size encoding visual_encoding data_encoding box layout
overall_layout layout_structure pattern data_pattern connection link grid cell region
data_area background title description axis_label label labels text_labels text text_content
words annotation number numbers legend color_code axes x_axis y_axis visualization chart
chart_structure graph image map data information knowledge concept detail""".split())

unmapped = [(n, ct) for n, ct in c.most_common() if n not in MAPPED]
mapped_ct = sum(ct for n, ct in c.most_common() if n in MAPPED)

print(f'Mapped: {mapped_ct}/{len(all_names)} ({mapped_ct*100//len(all_names)}%)')
print(f'Unmapped: {sum(ct for _, ct in unmapped)}/{len(all_names)} ({sum(ct for _, ct in unmapped)*100//len(all_names)}%)')
print(f'\nTop 15 still unmapped:')
for n, ct in unmapped[:15]:
    print(f'  {n}: {ct}')
