"""
add_oar_B_objects_column.py
----------------------------
Adds oar_B_synset_objects column to ResultsStepByStep_4.0.imageDataCompiled.csv:

  oar_B_synset_objects — synset only (deduplicated):
                         "subcategory (category)", e.g. "legend (furniture)"

The synsets are derived from the OAR-B extraction's objects column.

Sources:
  - vc_genome_output_full/vistype_profile/oar_image_traceability.csv
      objects column format: "raw_name --> category.subcategory [region] | ..."
  - comment_process/ResultsStepByStep_4.0.imageDataCompiled.csv  (modified in-place)
"""

import re
import argparse
import pandas as pd
from pathlib import Path


TRACEABILITY_PATH = Path("vc_genome_output_full/vistype_profile/oar_image_traceability.csv")
MAIN_CSV_PATH     = Path("comment_process/ResultsStepByStep_4.0.imageDataCompiled.csv")

# Regex: "raw_name --> category.subcategory [region]"
OBJ_PATTERN = re.compile(r"(.+?)\s*-->\s*(\w+)\.(\w+)\s*\[.+?\]")


def parse_objects(obj_str: str) -> str:
    """
    Parse one cell from the objects column and return formatted synset string.

    Input:  "legend --> furniture.legend [legend] | title --> text.title [title]"
    Output: "legend (furniture); title (text)"
    """
    if not isinstance(obj_str, str) or not obj_str.strip():
        return ""

    seen = []
    seen_set = set()
    for part in obj_str.split(" | "):
        m = OBJ_PATTERN.match(part.strip())
        if m:
            category, subcategory = m.group(2), m.group(3)
            token = f"{subcategory} ({category})"
            if token not in seen_set:
                seen.append(token)
                seen_set.add(token)

    return "; ".join(seen)


def main():
    parser = argparse.ArgumentParser(description="Add oar_B_synset_objects column to main CSV")
    parser.add_argument("--main-csv",     default=str(MAIN_CSV_PATH),     help="Path to main CSV")
    parser.add_argument("--traceability", default=str(TRACEABILITY_PATH), help="Path to oar_image_traceability.csv")
    args = parser.parse_args()

    main_csv     = pd.read_csv(args.main_csv)
    traceability = pd.read_csv(args.traceability)

    # Build per-image lookup
    syn_lookup = dict(zip(traceability["imageName"], traceability["objects"].apply(parse_objects)))

    # Drop old column if present, then add synset column
    if "oar_B_synset_objects" in main_csv.columns:
        main_csv = main_csv.drop(columns=["oar_B_synset_objects"])

    main_csv["oar_B_synset_objects"] = main_csv["imageName"].map(syn_lookup).fillna("")

    matched = main_csv["oar_B_synset_objects"].ne("").sum()
    print(f"Column 'oar_B_synset_objects' added:")
    print(f"  Matched images   : {matched}")
    print(f"  Unmatched images : {main_csv['oar_B_synset_objects'].eq('').sum()}")

    print("\nSample rows:")
    sample = main_csv[main_csv["oar_B_synset_objects"].ne("")][["imageName", "oar_B_synset_objects"]].head(5)
    for _, row in sample.iterrows():
        print(f"  {row['imageName']}: {row['oar_B_synset_objects']}")

    main_csv.to_csv(args.main_csv, index=False)
    print(f"\nSaved: {args.main_csv}")


if __name__ == "__main__":
    main()
