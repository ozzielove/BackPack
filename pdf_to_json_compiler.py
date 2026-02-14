"""High-performance PDF -> JSON compiler for bank statement extraction.

Compiler flow:
    PDF bytes -> page-wise extraction IR -> deterministic JSON artifacts

Architecture
------------
1) `PDFCompiler` (engine)
   - Incremental page extraction with `pdfplumber`.
   - OCR fallback only for low-density pages.
   - Optional bank-specific semantic parsing overlays.

2) Worker pipeline
   - File-level multiprocessing via `ProcessPoolExecutor`.
   - Per-file failure isolation with structured error reports.

3) CLI
   - Batch compilation from input directory to output directory.
   - Optional lockstep mode for sequential processing.

Performance notes
-----------------
- Pages are processed incrementally to avoid whole-document buffering.
- OCR rasterization is one-page-at-a-time.
- Multiprocessing parallelizes independent PDFs when enabled.

Customization
-------------
This module includes custom semantic parsers for:
- Navy Federal statements
- TD Bank statements

Semantic parsing is optional (`--emit-transactions`) so this module can operate as a
pure compiler IR backend when desired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None

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

    posting_date: str | None  # ISO date YYYY-MM-DD
    description: str
    amount: float | None  # normalized: money out negative, money in positive
    balance: float | None
    category: str | None
    source_bank: str


@dataclass(slots=True)
class BankProfile:
    """Detected bank identity + extracted summary metadata."""

    bank_name: str
    statement_period: str | None
    account_holder: str | None
    account_last4: str | None
    account_hash: str | None
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
    bank_profile: BankProfile | None
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


@dataclass(slots=True)
class StatementPeriod:
    start: date
    end: date


class PDFCompiler:
    """Compiles a single PDF into deterministic JSON IR with optional semantic parsing."""

    NAVY_MARKERS = ("navy federal", "statement of account", "access no")
    TD_MARKERS = ("td bank", "statement of account", "cust ref #", "primary account #")

    def __init__(
        self,
        text_density_threshold: int = 80,
        ocr_dpi: int = 300,
        custom_table_extractor: CustomTableExtractor | None = None,
        custom_page_processor: CustomPageProcessor | None = None,
        emit_transactions: bool = True,
    ) -> None:
        self.text_density_threshold = text_density_threshold
        self.ocr_dpi = ocr_dpi
        self.custom_table_extractor = custom_table_extractor
        self.custom_page_processor = custom_page_processor
        self.emit_transactions = emit_transactions

    def extract_pdf(self, pdf_path: Path) -> DocumentIR | ErrorReport:
        start = _utc_now()
        t0 = time.perf_counter()
        notes: list[str] = []
        pages_ir: list[PageIR] = []
        ocr_pages = 0
        native_pages = 0

        if pdfplumber is None:
            return ErrorReport(
                source_pdf=str(pdf_path),
                status="error",
                extraction_started_at=start,
                extraction_completed_at=_utc_now(),
                duration_seconds=0.0,
                error_type="MissingDependency",
                error_message="pdfplumber is not installed.",
                traceback="",
                extraction_notes=["Install pdfplumber to run extraction."],
            )

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                page_count = len(pdf.pages)
                combined_page_text: list[str] = []

                for idx, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ""
                    tables = self._extract_tables(page)

                    ocr_text: str | None = None
                    if self._needs_ocr(page_text, tables):
                        ocr_text, ocr_note = self._ocr_page(pdf_path, idx)
                        if ocr_note:
                            notes.append(f"Page {idx}: {ocr_note}")
                        if ocr_text:
                            ocr_pages += 1
                    else:
                        native_pages += 1

                    combined_page_text.append(f"{page_text}\n{ocr_text or ''}".strip())

                    page_ir = PageIR(
                        page_number=idx,
                        text=page_text,
                        tables=tables,
                        ocr_text=ocr_text,
                        extraction_confidence=self._confidence_score(page_text, ocr_text, tables),
                    )
                    if self.custom_page_processor is not None:
                        page_ir = self.custom_page_processor(page_ir)
                    pages_ir.append(page_ir)

            bank_profile: BankProfile | None = None
            transactions: list[TransactionRecord] = []
            if self.emit_transactions:
                full_text = "\n".join(combined_page_text)
                bank_profile, transactions, bank_notes = self._bank_specific_extract(full_text, pages_ir)
                notes.extend(bank_notes)
            else:
                notes.append("Semantic bank parsing disabled (--emit-transactions not set).")

            duration = time.perf_counter() - t0
            return DocumentIR(
                source_pdf=str(pdf_path),
                page_count=page_count,
                pages=pages_ir,
                extraction_notes=notes,
                extraction_started_at=start,
                extraction_completed_at=_utc_now(),
                metrics=DocumentMetrics(
                    duration_seconds=round(duration, 4),
                    ocr_pages=ocr_pages,
                    native_pages=native_pages,
                ),
                bank_profile=bank_profile,
                transactions=transactions,
            )
        except Exception as exc:
            duration = time.perf_counter() - t0
            return ErrorReport(
                source_pdf=str(pdf_path),
                status="error",
                extraction_started_at=start,
                extraction_completed_at=_utc_now(),
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
        text_len = len(text.strip())
        if text_len >= self.text_density_threshold:
            return False

        table_cells = sum(len(row) for table in tables for row in table)
        if table_cells > 0:
            return False

        # Avoid unnecessary OCR when page already shows likely transaction patterns.
        if re.search(r"\b\d{2}[/-]\d{2}(?:[/-]\d{2,4})?\b", text):
            return False

        return True

    def _ocr_page(self, pdf_path: Path, page_number: int) -> tuple[str | None, str | None]:
        if convert_from_path is None or pytesseract is None:
            return None, "OCR unavailable (missing pdf2image/pytesseract)."
        try:
            images = convert_from_path(
                pdf_path=str(pdf_path),
                dpi=self.ocr_dpi,
                first_page=page_number,
                last_page=page_number,
                thread_count=1,
            )
            if not images:
                return None, "OCR rasterization produced no images."
            text = pytesseract.image_to_string(images[0]).strip()
            if not text:
                return None, "OCR ran but returned empty text."
            return text, "OCR fallback used due to low text density."
        except Exception as exc:
            return None, f"OCR failed ({type(exc).__name__}): {exc}"

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

    def _bank_specific_extract(
        self, full_text: str, pages_ir: list[PageIR]
    ) -> tuple[BankProfile | None, list[TransactionRecord], list[str]]:
        navy_score = self._bank_score(full_text, self.NAVY_MARKERS)
        td_score = self._bank_score(full_text, self.TD_MARKERS)

        notes: list[str] = [f"Bank marker scores -> NAVY_FEDERAL={navy_score}, TD_BANK={td_score}"]

        if navy_score >= 2 and navy_score >= td_score:
            profile, txns, subnotes = self._extract_navy_federal(full_text, pages_ir)
            notes.extend(subnotes)
            notes.append("Detected bank profile: Navy Federal Credit Union.")
            return profile, txns, notes
        if td_score >= 2 and td_score > navy_score:
            profile, txns, subnotes = self._extract_td_bank(full_text, pages_ir)
            notes.extend(subnotes)
            notes.append("Detected bank profile: TD Bank.")
            return profile, txns, notes

        notes.append("Bank-specific parser not matched; metadata unavailable.")
        return None, [], notes

    @staticmethod
    def _bank_score(text: str, markers: tuple[str, ...]) -> int:
        lowered = text.lower()
        return sum(1 for marker in markers if marker in lowered)

    def _extract_navy_federal(
        self, full_text: str, pages_ir: list[PageIR]
    ) -> tuple[BankProfile, list[TransactionRecord], list[str]]:
        notes: list[str] = []
        period_raw = self._regex_group(full_text, r"Statement\s+Period\s*\n?\s*([0-9/\-\s]+)")
        period = _parse_statement_period(period_raw)
        access_no = self._regex_group(full_text, r"Access\s+No\.?\s*([A-Za-z0-9\-]+)")
        routing = self._regex_group(full_text, r"Routing\s+Number\s*[:\-]?\s*([\d\-]+)")
        account_raw = self._regex_group(full_text, r"EveryDay\s+Checking\s*-\s*([0-9]+)")

        holder = None
        holder_match = re.search(r"For\s+([A-Z][A-Z\s\.-]+)", full_text)
        if holder_match:
            holder = holder_match.group(1).strip()

        profile = BankProfile(
            bank_name="NAVY_FEDERAL",
            statement_period=period_raw,
            account_holder=holder,
            account_last4=_mask_last4(account_raw),
            account_hash=_hash_identifier(account_raw),
            routing_number=routing,
            access_reference=access_no,
        )

        txns = self._parse_navy_transactions_from_tables(pages_ir, period)
        if not txns:
            txns = self._parse_navy_transactions_from_text(full_text, period)
            notes.append("Navy Federal parser used text fallback (tables not usable).")
        notes.append(f"Navy Federal transaction rows parsed: {len(txns)}")
        return profile, txns, notes

    def _extract_td_bank(
        self, full_text: str, pages_ir: list[PageIR]
    ) -> tuple[BankProfile, list[TransactionRecord], list[str]]:
        notes: list[str] = []
        period_raw = self._regex_group(full_text, r"Statement\s+Period\s*:\s*([^\n]+)")
        period = _parse_statement_period(period_raw)
        account_raw = self._regex_group(full_text, r"Primary\s+Account\s*#\s*:\s*([\d\-]+)")
        cust_ref = self._regex_group(full_text, r"Cust\s+Ref\s*#\s*:\s*([^\n]+)")

        holder = None
        lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            if ln.upper() == "STATEMENT OF ACCOUNT" and i + 1 < len(lines):
                holder = lines[i + 1]
                break

        profile = BankProfile(
            bank_name="TD_BANK",
            statement_period=period_raw,
            account_holder=holder,
            account_last4=_mask_last4(account_raw),
            account_hash=_hash_identifier(account_raw),
            routing_number=None,
            access_reference=cust_ref,
        )

        txns = self._parse_td_transactions_from_tables(pages_ir, period)
        if not txns:
            txns = self._parse_td_transactions_from_text(full_text, period)
            notes.append("TD parser used text fallback (tables not usable).")
        notes.append(f"TD Bank transaction rows parsed: {len(txns)}")
        return profile, txns, notes

    def _parse_navy_transactions_from_tables(
        self, pages_ir: list[PageIR], period: StatementPeriod | None
    ) -> list[TransactionRecord]:
        txns: list[TransactionRecord] = []
        date_re = re.compile(r"^\d{2}-\d{2}$")
        money_re = re.compile(r"^-?\d{1,3}(?:,\d{3})*\.\d{2}$")

        for page in pages_ir:
            for table in page.tables:
                for row in table:
                    clean = [((c or "").strip()) for c in row]
                    if not any(clean):
                        continue
                    first = clean[0] if clean else ""
                    if not date_re.match(first):
                        continue
                    desc = " ".join(c for c in clean[1:-2] if c).strip() or (clean[1] if len(clean) > 1 else "")
                    amt_str = next((c for c in reversed(clean) if money_re.match(c)), None)
                    bal = None
                    amount = _safe_float(amt_str)
                    if len(clean) >= 2 and money_re.match(clean[-1]) and money_re.match(clean[-2]):
                        amount = _safe_float(clean[-2])
                        bal = _safe_float(clean[-1])
                    posting_iso = _resolve_mmdd(first, period)
                    txns.append(
                        TransactionRecord(
                            posting_date=posting_iso,
                            description=desc,
                            amount=self._normalize_amount(amount, desc),
                            balance=bal,
                            category=self._infer_category(desc),
                            source_bank="NAVY_FEDERAL",
                        )
                    )
        return txns

    def _parse_navy_transactions_from_text(
        self, full_text: str, period: StatementPeriod | None
    ) -> list[TransactionRecord]:
        txns: list[TransactionRecord] = []
        line_pattern = re.compile(
            r"(?P<date>\d{2}-\d{2})\s+(?P<desc>.+?)\s+(?P<amount>-?\d+\.\d{2})(?:\s+(?P<balance>-?\d+\.\d{2}))?$"
        )
        for raw in full_text.splitlines():
            line = " ".join(raw.split())
            m = line_pattern.search(line)
            if not m:
                continue
            desc = m.group("desc")
            amount = _safe_float(m.group("amount"))
            txns.append(
                TransactionRecord(
                    posting_date=_resolve_mmdd(m.group("date"), period),
                    description=desc,
                    amount=self._normalize_amount(amount, desc),
                    balance=_safe_float(m.group("balance")),
                    category=self._infer_category(desc),
                    source_bank="NAVY_FEDERAL",
                )
            )
        return txns

    def _parse_td_transactions_from_tables(
        self, pages_ir: list[PageIR], period: StatementPeriod | None
    ) -> list[TransactionRecord]:
        txns: list[TransactionRecord] = []
        date_re = re.compile(r"^\d{2}/\d{2}$")
        money_re = re.compile(r"^-?\d{1,3}(?:,\d{3})*\.\d{2}$")

        for page in pages_ir:
            for table in page.tables:
                for row in table:
                    clean = [((c or "").strip()) for c in row]
                    if not clean or not date_re.match(clean[0]):
                        continue
                    amounts = [c for c in clean if money_re.match(c)]
                    amount = _safe_float(amounts[-1]) if amounts else None
                    desc_parts = [c for c in clean[1:] if c and not money_re.match(c)]
                    desc = " ".join(desc_parts).strip()
                    if not desc:
                        continue
                    txns.append(
                        TransactionRecord(
                            posting_date=_resolve_mmdd(clean[0], period),
                            description=desc,
                            amount=self._normalize_amount(amount, desc),
                            balance=None,
                            category=self._infer_category(desc),
                            source_bank="TD_BANK",
                        )
                    )
        return txns

    def _parse_td_transactions_from_text(
        self, full_text: str, period: StatementPeriod | None
    ) -> list[TransactionRecord]:
        txns: list[TransactionRecord] = []
        lines = [" ".join(ln.split()) for ln in full_text.splitlines() if ln.strip()]

        date_line = re.compile(r"^(\d{2}/\d{2})\s+(.+)$")
        amount_only = re.compile(r"^-?\d{1,3}(?:,\d{3})*\.\d{2}$")

        i = 0
        while i < len(lines):
            m = date_line.match(lines[i])
            if not m:
                i += 1
                continue

            mmdd = m.group(1)
            desc_parts = [m.group(2)]
            j = i + 1
            amount: float | None = None

            while j < len(lines):
                if date_line.match(lines[j]):
                    break
                if amount_only.match(lines[j]):
                    amount = _safe_float(lines[j])
                    j += 1
                    break
                desc_parts.append(lines[j])
                j += 1

            if amount is not None:
                desc = " ".join(desc_parts).strip()
                txns.append(
                    TransactionRecord(
                        posting_date=_resolve_mmdd(mmdd, period),
                        description=desc,
                        amount=self._normalize_amount(amount, desc),
                        balance=None,
                        category=self._infer_category(desc),
                        source_bank="TD_BANK",
                    )
                )
            i = j if j > i else i + 1

        return txns

    @staticmethod
    def _normalize_amount(amount: float | None, description: str) -> float | None:
        if amount is None:
            return None
        lower = description.lower()
        if any(tok in lower for tok in ("deposit", "credit", "edi paymnt")):
            return abs(amount)
        if any(tok in lower for tok in ("payment", "debit", "withdraw", "purchase", "pos", "dbcrd", "pmt")):
            return -abs(amount)
        return amount

    @staticmethod
    def _infer_category(description: str) -> str | None:
        d = description.lower()
        if any(tok in d for tok in ("ach", "deposit", "edi paymnt", "credit")):
            return "deposit"
        if any(tok in d for tok in ("atm", "withdraw")):
            return "cash_withdrawal"
        if any(tok in d for tok in ("debit", "dbcrd", "purchase", "pos", "pmt", "paypal", "bill")):
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


def _mask_last4(identifier: str | None) -> str | None:
    if not identifier:
        return None
    digits = "".join(ch for ch in identifier if ch.isdigit())
    if not digits:
        return None
    return digits[-4:]


def _hash_identifier(identifier: str | None) -> str | None:
    if not identifier:
        return None
    clean = "".join(ch for ch in identifier if ch.isdigit())
    if not clean:
        return None
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()[:16]


def _parse_statement_period(raw: str | None) -> StatementPeriod | None:
    if not raw:
        return None

    # Supports forms like:
    # - 03/26/25 - 04/27/25
    # - Dec 12 2024-Jan 11 2025
    value = " ".join(raw.split())

    m = re.search(r"(\d{2}/\d{2}/\d{2,4})\s*-\s*(\d{2}/\d{2}/\d{2,4})", value)
    if m:
        start = _parse_date_flexible(m.group(1))
        end = _parse_date_flexible(m.group(2))
        if start and end:
            return StatementPeriod(start=start, end=end)

    m2 = re.search(
        r"([A-Za-z]{3}\s+\d{1,2}\s+\d{4})\s*-\s*([A-Za-z]{3}\s+\d{1,2}\s+\d{4})",
        value,
    )
    if m2:
        start = _parse_date_flexible(m2.group(1))
        end = _parse_date_flexible(m2.group(2))
        if start and end:
            return StatementPeriod(start=start, end=end)

    return None


def _parse_date_flexible(raw: str) -> date | None:
    for fmt in ("%m/%d/%y", "%m/%d/%Y", "%b %d %Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _resolve_mmdd(mmdd: str, period: StatementPeriod | None) -> str | None:
    m = re.match(r"^(\d{2})[/-](\d{2})$", mmdd)
    if not m:
        return None
    month = int(m.group(1))
    day = int(m.group(2))

    if period is None:
        return None

    candidate_years = {period.start.year, period.end.year, period.start.year - 1, period.end.year + 1}
    candidates: list[date] = []
    for year in candidate_years:
        try:
            candidates.append(date(year, month, day))
        except ValueError:
            continue

    in_range = [d for d in candidates if period.start <= d <= period.end]
    if in_range:
        chosen = min(in_range, key=lambda d: abs((period.end - d).days))
        return chosen.isoformat()

    if candidates:
        chosen = min(candidates, key=lambda d: abs((period.end - d).days))
        return chosen.isoformat()

    return None


def _iter_pdfs(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() == ".pdf":
            yield path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def _worker_compile(
    pdf_path: str,
    output_dir: str,
    text_density_threshold: int,
    ocr_dpi: int,
    emit_transactions: bool,
) -> WorkerResult:
    compiler = PDFCompiler(
        text_density_threshold=text_density_threshold,
        ocr_dpi=ocr_dpi,
        emit_transactions=emit_transactions,
    )
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
    lockstep: bool = False,
    emit_transactions: bool = True,
) -> tuple[int, int, float]:
    pdf_files = list(_iter_pdfs(input_dir))
    if not pdf_files:
        print(f"[INFO] No PDFs found in {input_dir}")
        return 0, 0, 0.0

    start = time.perf_counter()
    success = 0
    durations: list[float] = []
    total = len(pdf_files)

    if lockstep:
        print(f"[INFO] Lockstep mode enabled: processing {total} PDF(s) sequentially")
        for idx, pdf in enumerate(pdf_files, start=1):
            result = _worker_compile(str(pdf), str(output_dir), text_density_threshold, ocr_dpi, emit_transactions)
            durations.append(result.duration_seconds)
            if result.success:
                success += 1
                print(f"[OK] {idx}/{total} {pdf.name} -> {Path(result.output_json).name}")
            else:
                print(f"[ERR] {idx}/{total} {pdf.name} -> error report JSON emitted")
    else:
        worker_count = max(1, workers)
        print(f"[INFO] Compiling {total} PDF(s) with {worker_count} worker(s)")
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            future_map = {
                pool.submit(
                    _worker_compile,
                    str(p),
                    str(output_dir),
                    text_density_threshold,
                    ocr_dpi,
                    emit_transactions,
                ): p
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
                        print(f"[OK] {completed}/{total} {src.name} -> {Path(result.output_json).name}")
                    else:
                        print(f"[ERR] {completed}/{total} {src.name} -> error report JSON emitted")
                except Exception as exc:
                    print(f"[ERR] {completed}/{total} {src.name} -> worker crashed: {exc}")

    elapsed = time.perf_counter() - start
    failed = total - success
    avg = (sum(durations) / len(durations)) if durations else 0.0
    print("[SUMMARY] Compilation complete")
    print(f"[SUMMARY] Success={success} Failed={failed} Total={total}")
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
        "--lockstep",
        action="store_true",
        help="Process PDFs sequentially (overrides multiprocessing).",
    )
    parser.add_argument(
        "--text-density-threshold",
        type=int,
        default=80,
        help="Minimum native chars to skip OCR fallback",
    )
    parser.add_argument("--ocr-dpi", type=int, default=300, help="OCR rasterization DPI")
    parser.add_argument(
        "--emit-transactions",
        action="store_true",
        help="Enable bank-specific semantic parsing and transaction emission.",
    )
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
        lockstep=args.lockstep,
        emit_transactions=args.emit_transactions,
    )


if __name__ == "__main__":
    main()
