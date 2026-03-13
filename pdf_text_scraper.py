#!/usr/bin/env python3
"""Extract all text from one or more PDF files."""

from __future__ import annotations

import argparse
from pathlib import Path


def _import_fitz():
    try:
        import fitz  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyMuPDF is required. Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc
    return fitz


def extract_text_from_pdf(pdf_path: Path) -> str:
    fitz = _import_fitz()
    pages: list[str] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            page_text = page.get_text("text")
            pages.append(f"\n===== Page {i} =====\n{page_text}")
    return "\n".join(pages).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract text from PDF files and write one .txt file per PDF.",
    )
    parser.add_argument("pdfs", nargs="+", help="Path(s) to PDF files.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="extracted_text",
        help="Directory for output text files (default: extracted_text).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf in args.pdfs:
        pdf_path = Path(pdf)
        if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
            print(f"[skip] Not a valid PDF: {pdf_path}")
            continue

        text = extract_text_from_pdf(pdf_path)
        out_path = output_dir / f"{pdf_path.stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"[ok] Extracted text: {pdf_path} -> {out_path}")


if __name__ == "__main__":
    main()
