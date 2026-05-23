"""
Convert a PDF file to plain text using PyMuPDF (fitz).
Usage: python pdf_to_text.py <path_to_pdf> [output_txt]
If output_txt is omitted, prints to stdout.
"""

import sys
import fitz  # PyMuPDF


def pdf_to_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_text.py <path_to_pdf> [output_txt]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    text = pdf_to_text(pdf_path)

    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Text written to {out_path}")
    else:
        print(text)
