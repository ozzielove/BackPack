#!/usr/bin/env python3
"""Redact compensation/pay-history content from payroll PDFs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SENSITIVE_KEYWORDS = [
    "wages",
    "wage",
    "salary",
    "earnings",
    "gross pay",
    "net pay",
    "net check",
    "deductions",
    "benefits",
    "withheld",
    "withholding",
    "compensation",
    "pay history",
    "pay statement",
    "earnings statement",
    "federal income tax",
    "social security tax",
    "medicare tax",
    "state income tax",
    "local income tax",
    "tips",
    "overtime",
    "holiday work",
    "rate",
    "hours",
    "year to date",
    "ytd",
    "box 1",
    "box 3",
    "box 5",
    "box 16",
]

MONEY_RE = re.compile(r"^\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})$|^\$?\d+\.\d{2}$")
PERCENT_RE = re.compile(r"^\d+(?:\.\d+)?%$")
KEYWORD_RE = re.compile(r"|".join(re.escape(k) for k in SENSITIVE_KEYWORDS), re.IGNORECASE)


def _import_fitz():
    try:
        import fitz  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyMuPDF is required. Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc
    return fitz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Redact salary, wages, earnings, benefits, and pay-history data from PDFs.",
    )
    parser.add_argument("pdfs", nargs="+", help="Path(s) to PDF files")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="redacted_pdfs",
        help="Directory to write redacted PDFs (default: redacted_pdfs)",
    )
    return parser


def redact_pdf(in_path: Path, out_path: Path) -> int:
    fitz = _import_fitz()
    redaction_count = 0

    with fitz.open(in_path) as doc:
        for page in doc:
            blocks = page.get_text("blocks")
            for block in blocks:
                x0, y0, x1, y1, text, *_ = block
                if text and KEYWORD_RE.search(text):
                    page.add_redact_annot(
                        fitz.Rect(x0, y0, x1, y1),
                        text="REDACTED",
                        fill=(0, 0, 0),
                        text_color=(1, 1, 1),
                    )
                    redaction_count += 1

            words = page.get_text("words")
            for x0, y0, x1, y1, word, *_ in words:
                token = word.strip()
                if MONEY_RE.match(token) or PERCENT_RE.match(token):
                    page.add_redact_annot(
                        fitz.Rect(x0, y0, x1, y1),
                        fill=(0, 0, 0),
                    )
                    redaction_count += 1

            page.apply_redactions()

        doc.save(out_path, garbage=4, deflate=True)

    return redaction_count


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

        out_path = output_dir / f"{pdf_path.stem}_redacted.pdf"
        count = redact_pdf(pdf_path, out_path)
        print(f"[ok] Redacted {count} region(s): {pdf_path} -> {out_path}")


if __name__ == "__main__":
    main()
