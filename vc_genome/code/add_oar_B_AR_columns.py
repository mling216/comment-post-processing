"""
add_oar_B_AR_columns.py
------------------------
Adds two columns to the main image data CSV:

  oar_B_synset_attributes   — deduplicated "attribute_name (subcategory)" entries
                              e.g. "increases_perceived_complexity (legend); aids_quick_interpretation (title)"

  oar_B_synset_relationships — deduplicated "subcategory --predicate--> subcategory" entries
                              e.g. "legend --adds_complexity_relative_to--> title"

Source traceability column formats:
  attributes:    "attr_name @ raw_name (category.subcategory) [+/-] | ..."
  relationships: "raw_name(category.subcategory) --predicate--> raw_name(category.subcategory) [+/-] | ..."

Sources:
  - vc_genome_output_full/vistype_profile/oar_image_traceability.csv
  - comment_process/ResultsStepByStep - 4.0.imageDataCompiled.csv  (modified in-place)
"""

import re
import argparse
import pandas as pd
from pathlib import Path


TRACEABILITY_PATH = Path("vc_genome_output_full/vistype_profile/oar_image_traceability.csv")
MAIN_CSV_PATH     = Path("comment_process/ResultsStepByStep - 4.0.imageDataCompiled.csv")

# attr_name @ raw_name (category.subcategory) [+/-]
ATTR_PATTERN = re.compile(r"(\w+)\s*@\s*.+?\((\w+)\.(\w+)\)\s*\[[\+\-]\]")

# raw_name(category.subcategory) --predicate--> raw_name(category.subcategory) [+/-]
REL_PATTERN  = re.compile(r".+?\((\w+)\.(\w+)\)\s*--([\w_]+)-->\s*.+?\((\w+)\.(\w+)\)\s*\[[\+\-]\]")


def parse_attributes(attr_str: str) -> str:
    """
    Input:  "increases_perceived_complexity @ legend (furniture.legend) [+] | aids_quick_interpretation @ title (text.title) [-]"
    Output: "increases_perceived_complexity (legend); aids_quick_interpretation (title)"
    """
    if not isinstance(attr_str, str) or not attr_str.strip():
        return ""

    seen, seen_set = [], set()
    for part in attr_str.split(" | "):
        m = ATTR_PATTERN.match(part.strip())
        if m:
            attr_name   = m.group(1)
            subcategory = m.group(3)
            token = f"{attr_name} ({subcategory})"
            if token not in seen_set:
                seen.append(token)
                seen_set.add(token)

    return "; ".join(seen)


def parse_relationships(rel_str: str) -> str:
    """
    Input:  "legend(furniture.legend) --adds_complexity_relative_to--> title(text.title) [+]"
    Output: "legend --adds_complexity_relative_to--> title"
    """
    if not isinstance(rel_str, str) or not rel_str.strip():
        return ""

    seen, seen_set = [], set()
    for part in rel_str.split(" | "):
        m = REL_PATTERN.match(part.strip())
        if m:
            sub1      = m.group(2)
            predicate = m.group(3)
            sub2      = m.group(5)
            token = f"{sub1} --{predicate}--> {sub2}"
            if token not in seen_set:
                seen.append(token)
                seen_set.add(token)

    return "; ".join(seen)


def main():
    parser = argparse.ArgumentParser(description="Add oar_B_synset_attributes and oar_B_synset_relationships columns")
    parser.add_argument("--main-csv",     default=str(MAIN_CSV_PATH),     help="Path to main CSV")
    parser.add_argument("--traceability", default=str(TRACEABILITY_PATH), help="Path to oar_image_traceability.csv")
    args = parser.parse_args()

    main_csv     = pd.read_csv(args.main_csv, encoding="cp1252")
    traceability = pd.read_csv(args.traceability)

    attr_lookup = dict(zip(traceability["imageName"], traceability["attributes"].apply(parse_attributes)))
    rel_lookup  = dict(zip(traceability["imageName"], traceability["relationships"].apply(parse_relationships)))

    for col in ["oar_B_synset_attributes", "oar_B_synset_relationships"]:
        if col in main_csv.columns:
            main_csv = main_csv.drop(columns=[col])

    main_csv["oar_B_synset_attributes"]    = main_csv["imageName"].map(attr_lookup).fillna("")
    main_csv["oar_B_synset_relationships"] = main_csv["imageName"].map(rel_lookup).fillna("")

    for col in ["oar_B_synset_attributes", "oar_B_synset_relationships"]:
        matched = main_csv[col].ne("").sum()
        print(f"Column '{col}': {matched} matched, {main_csv[col].eq('').sum()} unmatched")

    print("\nSample (attributes):")
    sample = main_csv[main_csv["oar_B_synset_attributes"].ne("")][["imageName", "oar_B_synset_attributes"]].head(3)
    for _, row in sample.iterrows():
        print(f"  {row['imageName']}: {row['oar_B_synset_attributes'][:120]}")

    print("\nSample (relationships):")
    sample = main_csv[main_csv["oar_B_synset_relationships"].ne("")][["imageName", "oar_B_synset_relationships"]].head(3)
    for _, row in sample.iterrows():
        print(f"  {row['imageName']}: {row['oar_B_synset_relationships'][:120]}")

    main_csv.to_csv(args.main_csv, index=False)
    print(f"\nSaved: {args.main_csv}")


if __name__ == "__main__":
    main()
