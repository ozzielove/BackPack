#!/usr/bin/env python3
"""
tex_pdf_builder.py

A deterministic LaTeX PDF builder with defensive checks and structured reporting.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------- Utility structures -----------------------------

@dataclass
class CommandResult:
    returncode: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool


class SubprocessRunner:
    """Wrapper to run subprocess commands for easier mocking/testing."""

    def run(self, args: Sequence[str], cwd: Optional[Path] = None, timeout: Optional[int] = None) -> CommandResult:
        try:
            proc = subprocess.run(
                args,
                cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )
            return CommandResult(proc.returncode, proc.stdout, proc.stderr, False)
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if exc.stdout is not None else ""
            stderr = exc.stderr if exc.stderr is not None else ""
            return CommandResult(None, stdout, stderr, True)
        except FileNotFoundError:
            return CommandResult(None, "", "Executable not found", False)


# ---------------------------- Core functionality -----------------------------

SUPPORTED_ENGINES = ["pdflatex", "xelatex", "lualatex"]
DEFAULT_TIMEOUT = 60


@dataclass
class EngineProbeResult:
    engine: str
    candidates: List[str]
    callable: List[str]
    failed: List[str]


@dataclass
class AttemptResult:
    engine: str
    engine_path: str
    return_code: Optional[int]
    timed_out: bool
    stdout_excerpt: str
    stderr_excerpt: str


@dataclass
class FileResult:
    file_name: str
    source_path: str
    success: bool
    pdf_output_path: Optional[str]
    engine_used: Optional[str]
    return_code: Optional[int]
    stderr_excerpt: str
    attempts: List[AttemptResult]


class LatexBuilder:
    def __init__(self, runner: Optional[SubprocessRunner] = None, logger: Optional[logging.Logger] = None):
        self.runner = runner or SubprocessRunner()
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def _is_writable(path: Path) -> bool:
        try:
            test_file = path / ".__tex_pdf_builder_write_test"
            test_file.write_text("test", encoding="utf-8")
            test_file.unlink()
            return True
        except Exception:
            return False

    def discover_tex_files(self, directory: Path) -> List[Path]:
        files = sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".tex"])
        self.logger.debug("Discovered %d .tex files", len(files))
        return files

    def preflight_checks(
        self,
        source_dir: Path,
        out_dir: Path,
        tex_files: List[Path],
        dry_run: bool,
    ) -> Tuple[List[EngineProbeResult], Dict[str, Path]]:
        if platform.system().lower().startswith("win"):
            raise SystemExit("Windows is not supported. Run in a Unix-like environment.")

        if not source_dir.exists() or not source_dir.is_dir():
            raise SystemExit(f"Input directory does not exist or is not a directory: {source_dir}")
        if not os.access(source_dir, os.R_OK):
            raise SystemExit(f"Input directory is not readable: {source_dir}")

        if not tex_files:
            raise SystemExit(f"No .tex files found in directory: {source_dir}")

        for tex in tex_files:
            if not tex.exists() or not tex.is_file():
                raise SystemExit(f".tex file is missing: {tex}")
            if not os.access(tex, os.R_OK):
                raise SystemExit(f".tex file is not readable: {tex}")

        if not out_dir.exists():
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                raise SystemExit(f"Failed to create output directory {out_dir}: {exc}")
        if not out_dir.is_dir():
            raise SystemExit(f"Output path is not a directory: {out_dir}")
        if not self._is_writable(out_dir):
            raise SystemExit(f"Output directory is not writable: {out_dir}")

        probe_results, callable_engines = self.probe_engines()
        if not callable_engines:
            raise SystemExit(self._format_engine_probe_failure(probe_results))
        return probe_results, callable_engines

    def _expand_tex_bin_path(self, value: str) -> List[Path]:
        parts = value.split(":") if ":" in value else [value]
        result: List[Path] = []
        for part in parts:
            p = Path(part).expanduser()
            if p.is_file():
                result.append(p)
            elif p.is_dir():
                result.append(p)
        return result

    def _candidate_paths_for_engine(self, engine: str) -> List[Path]:
        candidates: List[Path] = []

        tex_bin_env = os.environ.get("TEX_BIN_PATH")
        if tex_bin_env:
            for entry in self._expand_tex_bin_path(tex_bin_env):
                if entry.is_file():
                    candidates.append(entry)
                elif entry.is_dir():
                    candidates.append(entry / engine)

        candidates.append(Path("/Library/TeX/texbin") / engine)

        for path_str in sorted(Path("/usr/local/texlive").glob("**/bin/*")):
            candidates.append(path_str / engine)

        # Deduplicate while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)
        return unique_candidates

    def _probe_engine_path(self, path: Path) -> Tuple[bool, str]:
        if not path.exists():
            return False, "not found"
        if not os.access(path, os.X_OK):
            return False, "not executable"

        result = self.runner.run([str(path), "--version"], timeout=5)
        if result.timed_out:
            return False, "timeout during --version"
        if result.returncode != 0:
            return False, f"non-zero exit ({result.returncode})"
        return True, "callable"

    @staticmethod
    def _format_engine_probe_failure(probe_results: List[EngineProbeResult]) -> str:
        search_paths: List[str] = []
        failure_details: List[str] = []
        for res in probe_results:
            search_paths.extend(res.candidates)
            if not res.callable:
                failure_details.extend(res.failed)
        if not search_paths:
            return "No callable LaTeX engines found. No candidate paths were discovered."
        message_lines = ["No callable LaTeX engines found.", "Searched candidate paths:"]
        for path in search_paths:
            message_lines.append(f" - {path}")
        if failure_details:
            message_lines.append("Failures:")
            for detail in failure_details:
                message_lines.append(f" - {detail}")
        else:
            message_lines.append("No engines were callable and no specific failure reasons were captured.")
        return "\n".join(message_lines)

    def probe_engines(self) -> Tuple[List[EngineProbeResult], Dict[str, Path]]:
        probe_results: List[EngineProbeResult] = []
        callable_paths: Dict[str, Path] = {}

        for engine in SUPPORTED_ENGINES:
            candidates = self._candidate_paths_for_engine(engine)
            callable_list: List[str] = []
            failed_list: List[str] = []
            for cand in candidates:
                success, reason = self._probe_engine_path(cand)
                if success:
                    callable_list.append(str(cand))
                    if engine not in callable_paths:
                        callable_paths[engine] = cand
                        self.logger.debug("Selected engine %s at %s", engine, cand)
                else:
                    failed_list.append(f"{cand}: {reason}")
            probe_results.append(
                EngineProbeResult(
                    engine=engine,
                    candidates=[str(c) for c in candidates],
                    callable=callable_list,
                    failed=failed_list,
                )
            )
        return probe_results, callable_paths

    def compile_file(
        self,
        tex_file: Path,
        output_dir: Path,
        engines: Dict[str, Path],
        timeout: int,
    ) -> FileResult:
        attempts: List[AttemptResult] = []
        success = False
        pdf_output_path: Optional[Path] = None
        chosen_engine: Optional[str] = None
        return_code: Optional[int] = None
        stderr_excerpt = ""

        tex_mtime = tex_file.stat().st_mtime
        for engine in SUPPORTED_ENGINES:
            engine_path = engines.get(engine)
            if not engine_path:
                continue
            cmd = [str(engine_path), "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "-no-shell-escape"]
            if output_dir != tex_file.parent:
                cmd.extend(["-output-directory", str(output_dir)])
            cmd.append(tex_file.name)

            self.logger.info("Compiling %s with %s", tex_file.name, engine)
            result = self.runner.run(cmd, cwd=tex_file.parent, timeout=timeout)
            stdout_excerpt = "\n".join(result.stdout.splitlines()[-50:])
            stderr_excerpt_local = "\n".join(result.stderr.splitlines()[-50:])

            attempts.append(
                AttemptResult(
                    engine=engine,
                    engine_path=str(engine_path),
                    return_code=result.returncode,
                    timed_out=result.timed_out,
                    stdout_excerpt=stdout_excerpt,
                    stderr_excerpt=stderr_excerpt_local,
                )
            )

            if result.timed_out:
                self.logger.error("Compilation timed out for %s using %s", tex_file, engine)
                continue

            if result.returncode != 0:
                self.logger.error("Compilation failed for %s using %s with return code %s", tex_file, engine, result.returncode)
                continue

            pdf_candidate = output_dir / (tex_file.stem + ".pdf")
            if self._validate_pdf(tex_file, pdf_candidate):
                success = True
                pdf_output_path = pdf_candidate
                chosen_engine = engine
                return_code = result.returncode
                stderr_excerpt = stderr_excerpt_local
                break
            else:
                self.logger.error("PDF validation failed for %s using %s", tex_file, engine)

        if not attempts:
            stderr_excerpt = "No callable LaTeX engines were available for this file."

        if not success and attempts:
            stderr_excerpt = attempts[-1].stderr_excerpt
        return FileResult(
            file_name=tex_file.name,
            source_path=str(tex_file),
            success=success,
            pdf_output_path=str(pdf_output_path) if pdf_output_path else None,
            engine_used=chosen_engine,
            return_code=return_code,
            stderr_excerpt=stderr_excerpt,
            attempts=attempts,
        )

    @staticmethod
    def _validate_pdf(tex_file: Path, pdf_file: Path) -> bool:
        if not pdf_file.exists() or not pdf_file.is_file():
            return False
        if pdf_file.stat().st_size <= 0:
            return False
        pdf_mtime = pdf_file.stat().st_mtime
        tex_mtime = tex_file.stat().st_mtime
        return pdf_mtime >= tex_mtime + 1.0


# ------------------------------ CLI handling ---------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic LaTeX PDF builder")
    parser.add_argument("--dir", dest="directory", default=os.getcwd(), help="Directory containing .tex files (default: cwd)")
    parser.add_argument("--out-dir", dest="out_dir", default=None, help="Output directory for PDFs")
    parser.add_argument("--json-report", dest="json_report", default=None, help="Path to write JSON build report")
    parser.add_argument("--timeout", dest="timeout", type=int, default=DEFAULT_TIMEOUT, help="Per-file compilation timeout in seconds (default: 60)")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Perform discovery and checks without compiling")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def setup_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("tex_pdf_builder")
    return logger


def human_summary(results: List[FileResult]) -> str:
    lines = []
    success_count = sum(1 for r in results if r.success)
    total = len(results)
    lines.append(f"Build results: {success_count}/{total} succeeded")
    for res in results:
        status = "OK" if res.success else "FAIL"
        engine = res.engine_used or "-"
        lines.append(f" - {res.file_name}: {status} (engine: {engine})")
    return "\n".join(lines)


def write_json_report(path: Path, report: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    except Exception as exc:
        raise SystemExit(f"Failed to write JSON report to {path}: {exc}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = setup_logging(args.verbose)

    source_dir = Path(args.directory).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else source_dir

    builder = LatexBuilder(logger=logger)
    tex_files = builder.discover_tex_files(source_dir)

    probe_results, callable_engines = builder.preflight_checks(source_dir, out_dir, tex_files, args.dry_run)

    results: List[FileResult] = []

    if args.dry_run:
        for tex in tex_files:
            results.append(
                FileResult(
                    file_name=tex.name,
                    source_path=str(tex),
                    success=True,
                    pdf_output_path=None,
                    engine_used=None,
                    return_code=None,
                    stderr_excerpt="",
                    attempts=[],
                )
            )
    else:
        for tex in tex_files:
            result = builder.compile_file(tex, out_dir, callable_engines, args.timeout)
            results.append(result)

    overall_success = all(r.success for r in results)

    report = {
        "build_metadata": {
            "script": "tex_pdf_builder.py",
            "python": platform.python_version(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dir": str(source_dir),
            "out_dir": str(out_dir),
            "timeout_seconds": args.timeout,
            "dry_run": bool(args.dry_run),
            "verbose": bool(args.verbose),
            "engines_probed": [
                {
                    "engine": r.engine,
                    "candidates": r.candidates,
                    "callable": r.callable,
                    "failed": r.failed,
                }
                for r in probe_results
            ],
        },
        "results": [
            {
                "file_name": r.file_name,
                "source_path": r.source_path,
                "success": r.success,
                "pdf_output_path": r.pdf_output_path,
                "engine_used": r.engine_used,
                "return_code": r.return_code,
                "stderr_excerpt": r.stderr_excerpt,
                "attempts": [
                    {
                        "engine": a.engine,
                        "engine_path": a.engine_path,
                        "return_code": a.return_code,
                        "timed_out": a.timed_out,
                        "stdout_excerpt": a.stdout_excerpt,
                        "stderr_excerpt": a.stderr_excerpt,
                    }
                    for a in r.attempts
                ],
            }
            for r in results
        ],
        "overall_success": overall_success,
    }

    if args.json_report:
        write_json_report(Path(args.json_report), report)

    print(human_summary(results))

    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
