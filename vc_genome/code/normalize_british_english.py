"""
normalize_british_english.py
-----------------------------
Replace British English spellings with American English equivalents
across all string columns in a CSV file.

Usage:
    python vc_genome/code/normalize_british_english.py [--csv PATH]
"""

import re
import argparse
import pandas as pd
from pathlib import Path

DEFAULT_CSV = Path("comment_process/ResultsStepByStep - 4.0.imageDataCompiledpfix.csv")

# -----------------------------------------------------------------------
# Replacement rules: (pattern, replacement)
# Patterns use word boundaries (\b) for precision.
# Order matters: longer/more-specific patterns first.
# -----------------------------------------------------------------------
RULES = [
    # -our → -or
    (r"\bcolours\b",        "colors"),
    (r"\bcolour\b",         "color"),
    (r"\bcoloured\b",       "colored"),
    (r"\bcolouring\b",      "coloring"),
    (r"\bcolourful\b",      "colorful"),
    (r"\bcolourfull\b",     "colorful"),   # common typo
    (r"\bcolourless\b",     "colorless"),
    (r"\bcolourings\b",     "colorings"),
    (r"\bmulticoloured\b",  "multicolored"),
    (r"\bmulticolour\b",    "multicolor"),
    (r"\bbehaviours\b",     "behaviors"),
    (r"\bbehaviour\b",      "behavior"),
    (r"\bfavours\b",        "favors"),
    (r"\bfavour\b",         "favor"),
    (r"\bhonours\b",        "honors"),
    (r"\bhonour\b",         "honor"),
    (r"\bneighbours\b",     "neighbors"),
    (r"\bneighbour\b",      "neighbor"),
    (r"\bhumours\b",        "humors"),
    (r"\bhumour\b",         "humor"),
    (r"\bflavours\b",       "flavors"),
    (r"\bflavour\b",        "flavor"),
    (r"\blabours\b",        "labors"),
    (r"\blabour\b",         "labor"),
    (r"\bramours\b",        "rumors"),
    (r"\bramour\b",         "rumor"),

    # -ise / -ising / -ised / -isation → -ize / -izing / -ized / -ization
    (r"\brealised\b",       "realized"),
    (r"\brealising\b",      "realizing"),
    (r"\brealisations\b",   "realizations"),
    (r"\brealisation\b",    "realization"),
    (r"\brealise\b",        "realize"),
    (r"\bdisorganised\b",   "disorganized"),
    (r"\bdisorganise\b",    "disorganize"),
    (r"\breorganised\b",    "reorganized"),
    (r"\breorganise\b",     "reorganize"),
    (r"\borganised\b",      "organized"),
    (r"\borganising\b",     "organizing"),
    (r"\borganisations\b",  "organizations"),
    (r"\borganisation\b",   "organization"),
    (r"\borganise\b",       "organize"),
    (r"\brecognised\b",     "recognized"),
    (r"\brecognising\b",    "recognizing"),
    (r"\brecognise\b",      "recognize"),
    (r"\banalysed\b",       "analyzed"),
    (r"\banalysing\b",      "analyzing"),
    (r"\banalyse\b",        "analyze"),
    (r"\bvisualised\b",     "visualized"),
    (r"\bvisualising\b",    "visualizing"),
    (r"\bvisualise\b",      "visualize"),
    (r"\bsummarised\b",     "summarized"),
    (r"\bsummarising\b",    "summarizing"),
    (r"\bsummarise\b",      "summarize"),
    (r"\bspecialised\b",    "specialized"),
    (r"\bspecialising\b",   "specializing"),
    (r"\bspecialise\b",     "specialize"),
    (r"\bstandardised\b",   "standardized"),
    (r"\bstandardising\b",  "standardizing"),
    (r"\bstandardise\b",    "standardize"),
    (r"\bcategorised\b",    "categorized"),
    (r"\bcategorising\b",   "categorizing"),
    (r"\bcategorise\b",     "categorize"),
    (r"\bcharacterised\b",  "characterized"),
    (r"\bcharacterising\b", "characterizing"),
    (r"\bcharacterise\b",   "characterize"),
    (r"\bprioritised\b",    "prioritized"),
    (r"\bprioritising\b",   "prioritizing"),
    (r"\bprioritise\b",     "prioritize"),
    (r"\bemphasised\b",     "emphasized"),
    (r"\bemphasising\b",    "emphasizing"),
    (r"\bemphasise\b",      "emphasize"),
    (r"\bminimised\b",      "minimized"),
    (r"\bminimising\b",     "minimizing"),
    (r"\bminimise\b",       "minimize"),
    (r"\bmaximised\b",      "maximized"),
    (r"\bmaximising\b",     "maximizing"),
    (r"\bmaximise\b",       "maximize"),
    (r"\butilised\b",       "utilized"),
    (r"\butilising\b",      "utilizing"),
    (r"\butilise\b",        "utilize"),

    # -re → -er
    (r"\bcentres\b",        "centers"),
    (r"\bcentre\b",         "center"),
    (r"\btheatres\b",       "theaters"),
    (r"\btheatre\b",        "theater"),
    (r"\bfibres\b",         "fibers"),
    (r"\bfibre\b",          "fiber"),
    (r"\bmetres\b",         "meters"),
    (r"\bmetre\b",          "meter"),
    (r"\bspectre\b",        "specter"),
    (r"\blitre\b",          "liter"),

    # double-l in -ing/-ed
    (r"\blabelling\b",      "labeling"),
    (r"\blabelled\b",       "labeled"),
    (r"\btravelling\b",     "traveling"),
    (r"\btravelled\b",      "traveled"),
    (r"\bmodelling\b",      "modeling"),
    (r"\bmodelled\b",       "modeled"),
    (r"\bcancelling\b",     "canceling"),
    (r"\bcancelled\b",      "canceled"),
    (r"\bfuelling\b",       "fueling"),
    (r"\bfuelled\b",        "fueled"),

    # -ogue → -og
    (r"\bcatalogues\b",     "catalogs"),
    (r"\bcatalogue\b",      "catalog"),
    (r"\bdialogues\b",      "dialogs"),
    (r"\bdialogue\b",       "dialog"),

    # -gramme → -gram
    (r"\bprogrammes\b",     "programs"),
    (r"\bprogramme\b",      "program"),

    # -ence → -ense
    (r"\bdefences\b",       "defenses"),
    (r"\bdefence\b",        "defense"),
    (r"\boffences\b",       "offenses"),
    (r"\boffence\b",        "offense"),
    (r"\blicences\b",       "licenses"),
    (r"\blicence\b",        "license"),
    (r"\bpretences\b",      "pretenses"),
    (r"\bpretence\b",       "pretense"),

    # -ise noun forms
    (r"\bpractises\b",      "practices"),
    (r"\bpractise\b",       "practice"),

    # grey → gray
    (r"\bgreys\b",          "grays"),
    (r"\bgrey\b",           "gray"),
    (r"\bgreyed\b",         "grayed"),
    (r"\bgreying\b",        "graying"),
]

# Compile with IGNORECASE; replacement preserves the matched case pattern
def _make_replacer(pattern: str, replacement: str):
    rx = re.compile(pattern, re.IGNORECASE)

    def _replace(m: re.Match) -> str:
        orig = m.group(0)
        # If all upper, return all upper; if title case, title case; else lower
        if orig.isupper():
            return replacement.upper()
        if orig.istitle():
            return replacement.capitalize()
        return replacement

    return rx, _replace


COMPILED = [_make_replacer(p, r) for p, r in RULES]


def normalize_cell(value: str) -> str:
    for rx, replacer in COMPILED:
        value = rx.sub(replacer, value)
    return value


def main():
    parser = argparse.ArgumentParser(description="Normalize British English spellings to American English in a CSV")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to CSV file (modified in-place)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    str_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]
    total_changes = 0

    for col in str_cols:
        original = df[col].copy()
        df[col] = df[col].apply(lambda x: normalize_cell(x) if isinstance(x, str) else x)
        n = (df[col] != original).sum()
        if n:
            print(f"  {col}: {n} cell(s) changed")
            total_changes += n

    print(f"\nTotal cells changed: {total_changes}")
    df.to_csv(args.csv, index=False)
    print(f"Saved: {args.csv}")


if __name__ == "__main__":
    main()
