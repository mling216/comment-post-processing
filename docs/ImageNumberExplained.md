# Image Number Explained

**707 unique images in sheet `1103808983` → 495 in your table**

## Breakdown

| Reason | Count | Details |
|--------|------:|---------|
| **Not in main image sheet** (`1390591889`) | **202** | These images appear in paired comparisons but were never in the 505-image study list. Crucially, **all 202 have zero curated phrases** — they were compared but nothing was annotated for them. |
| **Dropped by phrase reduction pipeline** | **10** | These ARE in the main 505-image sheet but have no entries in `image_phrase_matching.csv`, so the pipeline had nothing to compile for them. |
| **Total gap** | **212** | 707 − 212 = 495 ✓ |

## Minor Factors

- **ToRemove rows:** Only 13 rows (6 unique images) are flagged `ToRemove=1`, and all 6 are already among the 202 not-in-main-sheet images. So ToRemove is a negligible contributor.
- **No meaningful user comments:** The dominant reason (202/212 = **95%**) is that these images were in paired-comparison rows but had **no curated phrases**, and they weren't even in the main study image list.
- **Remaining 10 images** (e.g., `InfoVisC.211.3(2).png`, `InfoVisJ.2082.7.png`) are in the main list but got no phrase-matching entries during the pipeline.