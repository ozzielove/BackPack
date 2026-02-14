"""High-performance PDF -> JSON compiler for bank statement extraction.

Compiler flow:
    PDF bytes -> page-wise extraction IR -> deterministic JSON artifacts

Architecture
------------
1) `PDFCompiler` (engine)
   - Incremental page extraction with `pdfplumber`.
   - OCR fallback only for low-density pages.
   - Bank-profile detection + bank-specific parsing overlays.

2) Worker pipeline
   - File-level multiprocessing via `ProcessPoolExecutor`.
   - Per-file failure isolation with structured error reports.

3) CLI
   - Batch compilation from input directory to output directory.

Performance notes
-----------------
- Pages are processed incrementally to avoid whole-document buffering.
- OCR rasterization is one-page-at-a-time.
- Multiprocessing parallelizes independent PDFs.

Customization
-------------
This module includes custom extractors for:
- Navy Federal statements
- TD Bank statements

The bank-specific extraction emits normalized metadata and transaction rows that
are convenient for recurring-bill detectors and downstream parser agents.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pdfplumber

try:
    from pdf2image import convert_from_path
except Exception:  # pragma: no cover
    convert_from_path = None

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

ISO_TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


@dataclass(slots=True)
class TransactionRecord:
    """Normalized transaction row for downstream recurring-bill analysis."""

    posting_date: str | None
    description: str
    amount: float | None
    balance: float | None
    category: str | None
    source_bank: str


@dataclass(slots=True)
class BankProfile:
    """Detected bank identity + extracted summary metadata."""

    bank_name: str
    statement_period: str | None
    account_holder: str | None
    account_number_masked: str | None
    routing_number: str | None
    access_reference: str | None


@dataclass(slots=True)
class PageIR:
    page_number: int
    text: str
    tables: list[list[list[str | None]]]
    ocr_text: str | None
    extraction_confidence: float


@dataclass(slots=True)
class DocumentMetrics:
    duration_seconds: float
    ocr_pages: int
    native_pages: int


@dataclass(slots=True)
class DocumentIR:
    source_pdf: str
    page_count: int
    pages: list[PageIR]
    extraction_notes: list[str]
    extraction_started_at: str
    extraction_completed_at: str
    metrics: DocumentMetrics
    bank_profile: BankProfile
    transactions: list[TransactionRecord]


@dataclass(slots=True)
class ErrorReport:
    source_pdf: str
    status: str
    extraction_started_at: str
    extraction_completed_at: str
    duration_seconds: float
    error_type: str
    error_message: str
    traceback: str
    extraction_notes: list[str] = field(default_factory=list)


CustomTableExtractor = Callable[[Any], list[list[list[str | None]]]]
CustomPageProcessor = Callable[[PageIR], PageIR]


class PDFCompiler:
    """Compiles a single PDF into deterministic JSON IR with bank customizations."""

    NAVY_MARKERS = ("navy federal", "statement of account", "access no")
    TD_MARKERS = ("td bank", "statement of account", "cust ref #", "primary account #")

    def __init__(
        self,
        text_density_threshold: int = 80,
        ocr_dpi: int = 300,
        custom_table_extractor: CustomTableExtractor | None = None,
        custom_page_processor: CustomPageProcessor | None = None,
    ) -> None:
        self.text_density_threshold = text_density_threshold
        self.ocr_dpi = ocr_dpi
        self.custom_table_extractor = custom_table_extractor
        self.custom_page_processor = custom_page_processor

    def extract_pdf(self, pdf_path: Path) -> DocumentIR | ErrorReport:
        start = _utc_now()
        t0 = time.perf_counter()
        notes: list[str] = []
        pages_ir: list[PageIR] = []
        ocr_pages = 0
        native_pages = 0

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                page_count = len(pdf.pages)
                full_text_parts: list[str] = []

                for index, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ""
                    tables = self._extract_tables(page)

                    ocr_text: str | None = None
                    if self._needs_ocr(page_text, tables):
                        ocr_text = self._ocr_page(pdf_path, index)
                        if ocr_text:
                            notes.append(f"Page {index}: OCR fallback used due to low text density.")
                            ocr_pages += 1
                        else:
                            notes.append(f"Page {index}: OCR fallback attempted but yielded no text.")
                    else:
                        native_pages += 1

                    combined_text = f"{page_text}\n{ocr_text or ''}".strip()
                    full_text_parts.append(combined_text)

                    page_ir = PageIR(
                        page_number=index,
                        text=page_text,
                        tables=tables,
                        ocr_text=ocr_text,
                        extraction_confidence=self._confidence_score(page_text, ocr_text, tables),
                    )

                    if self.custom_page_processor is not None:
                        page_ir = self.custom_page_processor(page_ir)

                    pages_ir.append(page_ir)

            full_text = "\n".join(full_text_parts)
            bank_profile, transactions, bank_notes = self._bank_specific_extract(full_text)
            notes.extend(bank_notes)

            duration = time.perf_counter() - t0
            end = _utc_now()
            return DocumentIR(
                source_pdf=str(pdf_path),
                page_count=len(pages_ir),
                pages=pages_ir,
                extraction_notes=notes,
                extraction_started_at=start,
                extraction_completed_at=end,
                metrics=DocumentMetrics(
                    duration_seconds=round(duration, 4),
                    ocr_pages=ocr_pages,
                    native_pages=native_pages,
                ),
                bank_profile=bank_profile,
                transactions=transactions,
            )

        except Exception as exc:
            end = _utc_now()
            duration = time.perf_counter() - t0
            return ErrorReport(
                source_pdf=str(pdf_path),
                status="error",
                extraction_started_at=start,
                extraction_completed_at=end,
                duration_seconds=round(duration, 4),
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback=traceback.format_exc(),
                extraction_notes=["Compilation failed; worker continued."],
            )

    def _extract_tables(self, page: Any) -> list[list[list[str | None]]]:
        if self.custom_table_extractor is not None:
            return self.custom_table_extractor(page)
        rows = page.extract_tables() or []
        return [[[cell if cell is not None else None for cell in row] for row in table] for table in rows]

    def _needs_ocr(self, text: str, tables: list[list[list[str | None]]]) -> bool:
        if len(text.strip()) >= self.text_density_threshold:
            return False
        table_cells = sum(len(row) for table in tables for row in table)
        return table_cells == 0

    def _ocr_page(self, pdf_path: Path, page_number: int) -> str | None:
        if convert_from_path is None or pytesseract is None:
            return None
        try:
            images = convert_from_path(
                pdf_path=str(pdf_path),
                dpi=self.ocr_dpi,
                first_page=page_number,
                last_page=page_number,
                thread_count=1,
            )
            if not images:
                return None
            text = pytesseract.image_to_string(images[0])
            return text.strip() or None
        except Exception:
            return None

    @staticmethod
    def _confidence_score(text: str, ocr_text: str | None, tables: list[list[list[str | None]]]) -> float:
        native_chars = len(text.strip())
        ocr_chars = len((ocr_text or "").strip())
        table_bonus = min(0.25, 0.05 * len(tables))

        if native_chars >= 200:
            base = 0.92
        elif native_chars >= 80:
            base = 0.80
        elif ocr_chars >= 80:
            base = 0.72
        elif native_chars > 0 or ocr_chars > 0:
            base = 0.5
        else:
            base = 0.2
        return round(min(1.0, base + table_bonus), 3)

    def _bank_specific_extract(self, full_text: str) -> tuple[BankProfile, list[TransactionRecord], list[str]]:
        lowered = full_text.lower()
        if all(marker in lowered for marker in self.NAVY_MARKERS):
            profile, txns, notes = self._extract_navy_federal(full_text)
            notes.append("Detected bank profile: Navy Federal Credit Union.")
            return profile, txns, notes
        if all(marker in lowered for marker in self.TD_MARKERS):
            profile, txns, notes = self._extract_td_bank(full_text)
            notes.append("Detected bank profile: TD Bank.")
            return profile, txns, notes

        profile = BankProfile(
            bank_name="UNKNOWN",
            statement_period=self._regex_group(full_text, r"Statement\s+Period[:\s]+([^\n]+)"),
            account_holder=None,
            account_number_masked=self._regex_group(full_text, r"Account\s*(?:#|No\.?|Number)\s*[:\-]?\s*([\d\-*]+)"),
            routing_number=self._regex_group(full_text, r"Routing\s+Number\s*[:\-]?\s*([\d\-]+)"),
            access_reference=self._regex_group(full_text, r"Access\s+No\.?\s*[:\-]?\s*([A-Za-z0-9\-]+)"),
        )
        return profile, [], ["Bank-specific parser not matched; returned generic metadata only."]

    def _extract_navy_federal(self, full_text: str) -> tuple[BankProfile, list[TransactionRecord], list[str]]:
        notes: list[str] = []
        period = self._regex_group(full_text, r"Statement\s+Period\s*\n?\s*([0-9/\-\s]+)")
        access_no = self._regex_group(full_text, r"Access\s+No\.?\s*([A-Za-z0-9\-]+)")
        routing = self._regex_group(full_text, r"Routing\s+Number\s*[:\-]?\s*([\d\-]+)")
        acct = self._regex_group(full_text, r"EveryDay\s+Checking\s*-\s*([0-9]+)")

        holder = None
        holder_match = re.search(r"For\s+([A-Z][A-Z\s\.-]+)", full_text)
        if holder_match:
            holder = holder_match.group(1).strip()

        profile = BankProfile(
            bank_name="NAVY_FEDERAL",
            statement_period=period,
            account_holder=holder,
            account_number_masked=acct,
            routing_number=routing,
            access_reference=access_no,
        )

        txns = self._parse_navy_transactions(full_text)
        notes.append(f"Navy Federal transaction rows parsed: {len(txns)}")
        return profile, txns, notes

    def _extract_td_bank(self, full_text: str) -> tuple[BankProfile, list[TransactionRecord], list[str]]:
        notes: list[str] = []
        period = self._regex_group(full_text, r"Statement\s+Period\s*:\s*([^\n]+)")
        account = self._regex_group(full_text, r"Primary\s+Account\s*#\s*:\s*([\d\-]+)")
        cust_ref = self._regex_group(full_text, r"Cust\s+Ref\s*#\s*:\s*([^\n]+)")

        holder = None
        lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            if ln.upper() == "STATEMENT OF ACCOUNT" and i + 1 < len(lines):
                holder = lines[i + 1]
                break

        profile = BankProfile(
            bank_name="TD_BANK",
            statement_period=period,
            account_holder=holder,
            account_number_masked=account,
            routing_number=None,
            access_reference=cust_ref,
        )

        txns = self._parse_td_transactions(full_text)
        notes.append(f"TD Bank transaction rows parsed: {len(txns)}")
        return profile, txns, notes

    def _parse_navy_transactions(self, full_text: str) -> list[TransactionRecord]:
        txns: list[TransactionRecord] = []
        # Example pattern: 04-22 POS Debit- ... CA ... 29.95 26.10
        line_pattern = re.compile(
            r"(?P<date>\d{2}-\d{2})\s+(?P<desc>.+?)\s+(?P<amount>-?\d+\.\d{2})\s+(?P<balance>-?\d+\.\d{2})$"
        )
        for raw in full_text.splitlines():
            line = " ".join(raw.split())
            m = line_pattern.search(line)
            if not m:
                continue
            txns.append(
                TransactionRecord(
                    posting_date=m.group("date"),
                    description=m.group("desc"),
                    amount=_safe_float(m.group("amount")),
                    balance=_safe_float(m.group("balance")),
                    category=self._infer_category(m.group("desc")),
                    source_bank="NAVY_FEDERAL",
                )
            )
        return txns

    def _parse_td_transactions(self, full_text: str) -> list[TransactionRecord]:
        txns: list[TransactionRecord] = []
        lines = [" ".join(ln.split()) for ln in full_text.splitlines()]

        # Robust two-line/one-line matcher:
        #   12/23 DBCRD PUR AP, ...
        #   TESLA SUPERCHARGER ...
        #   6.98
        i = 0
        date_prefix = re.compile(r"^(\d{2}/\d{2})\s+(.+)$")
        amount_only = re.compile(r"^-?\d{1,3}(?:,\d{3})*\.\d{2}$")

        while i < len(lines):
            m = date_prefix.match(lines[i])
            if not m:
                i += 1
                continue

            posting_date = m.group(1)
            desc = m.group(2)
            amount: float | None = None

            if i + 1 < len(lines) and amount_only.match(lines[i + 1]):
                amount = _safe_float(lines[i + 1])
                i += 2
            elif i + 2 < len(lines) and amount_only.match(lines[i + 2]):
                desc = f"{desc} {lines[i + 1]}"
                amount = _safe_float(lines[i + 2])
                i += 3
            else:
                i += 1
                continue

            txns.append(
                TransactionRecord(
                    posting_date=posting_date,
                    description=desc,
                    amount=amount,
                    balance=None,
                    category=self._infer_category(desc),
                    source_bank="TD_BANK",
                )
            )

        return txns

    @staticmethod
    def _infer_category(description: str) -> str | None:
        d = description.lower()
        if any(tok in d for tok in ("ach", "deposit", "edi paymnt")):
            return "deposit"
        if any(tok in d for tok in ("atm", "withdraw")):
            return "cash_withdrawal"
        if any(tok in d for tok in ("debit", "dbcrd", "purchase", "pos", "pmt", "paypal")):
            return "payment"
        return None

    @staticmethod
    def _regex_group(text: str, pattern: str) -> str | None:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        return m.group(1).strip() if m else None


@dataclass(slots=True)
class WorkerResult:
    source_pdf: str
    output_json: str
    success: bool
    duration_seconds: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime(ISO_TS_FORMAT)


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def _iter_pdfs(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() == ".pdf":
            yield path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def _worker_compile(pdf_path: str, output_dir: str, text_density_threshold: int, ocr_dpi: int) -> WorkerResult:
    compiler = PDFCompiler(text_density_threshold=text_density_threshold, ocr_dpi=ocr_dpi)
    src = Path(pdf_path)
    out = Path(output_dir) / f"{src.stem}.json"

    result = compiler.extract_pdf(src)
    payload = asdict(result)
    _write_json(out, payload)

    duration = payload.get("metrics", {}).get("duration_seconds", payload.get("duration_seconds", 0.0))
    return WorkerResult(
        source_pdf=pdf_path,
        output_json=str(out),
        success=not isinstance(result, ErrorReport),
        duration_seconds=float(duration),
    )


def compile_directory(
    input_dir: Path,
    output_dir: Path,
    workers: int,
    text_density_threshold: int = 80,
    ocr_dpi: int = 300,
) -> tuple[int, int, float]:
    pdf_files = list(_iter_pdfs(input_dir))
    if not pdf_files:
        print(f"[INFO] No PDFs found in {input_dir}")
        return 0, 0, 0.0

    start = time.perf_counter()
    success = 0
    durations: list[float] = []

    worker_count = max(1, workers)
    print(f"[INFO] Compiling {len(pdf_files)} PDF(s) with {worker_count} worker(s)")

    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        future_map = {
            pool.submit(_worker_compile, str(p), str(output_dir), text_density_threshold, ocr_dpi): p
            for p in pdf_files
        }
        completed = 0
        for future in as_completed(future_map):
            completed += 1
            src = future_map[future]
            try:
                result = future.result()
                durations.append(result.duration_seconds)
                if result.success:
                    success += 1
                    print(f"[OK] {completed}/{len(pdf_files)} {src.name} -> {Path(result.output_json).name}")
                else:
                    print(f"[ERR] {completed}/{len(pdf_files)} {src.name} -> error report JSON emitted")
            except Exception as exc:
                print(f"[ERR] {completed}/{len(pdf_files)} {src.name} -> worker crashed: {exc}")

    elapsed = time.perf_counter() - start
    failed = len(pdf_files) - success
    avg = (sum(durations) / len(durations)) if durations else 0.0
    print("[SUMMARY] Compilation complete")
    print(f"[SUMMARY] Success={success} Failed={failed} Total={len(pdf_files)}")
    print(f"[SUMMARY] Wall={elapsed:.2f}s AvgFile={avg:.2f}s")
    return success, failed, elapsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile bank statement PDFs into JSON IR")
    parser.add_argument("--input", required=True, type=Path, help="Directory containing input PDFs")
    parser.add_argument("--output", required=True, type=Path, help="Directory for output JSON files")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of worker processes",
    )
    parser.add_argument(
        "--text-density-threshold",
        type=int,
        default=80,
        help="Minimum native chars to skip OCR fallback",
    )
    parser.add_argument("--ocr-dpi", type=int, default=300, help="OCR rasterization DPI")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.input.exists() or not args.input.is_dir():
        raise SystemExit(f"Input directory invalid: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    compile_directory(
        input_dir=args.input,
        output_dir=args.output,
        workers=args.workers,
        text_density_threshold=args.text_density_threshold,
        ocr_dpi=args.ocr_dpi,
    )


if __name__ == "__main__":
    main()
