#!/usr/bin/env python3
"""Internship pipeline gatekeeper with verification, SSRF guard, and persistence."""

import argparse
import doctest
import csv
import datetime as dt
import ipaddress
import json
import logging
import os
import re
import shutil
import socket
import sys
import tempfile
import urllib.parse
import urllib.request
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PAL_ROOT_DEFAULT = "/Users/ozirusmorency/Downloads/Pal"
NEED_TO_APPLY_DEFAULT = "/Users/ozirusmorency/Downloads/Pal/Need_To_Apply"
EXCLUSIONS_DEFAULT = "/Users/ozirusmorency/Downloads/Pal/Job_Search_System/search_exclusions.csv"
VERIFICATION_DEFAULT = "/Users/ozirusmorency/Downloads/Pal/Job_Search_System/internship_verification_list.csv"
TRACKER_DEFAULT = "/Users/ozirusmorency/Downloads/Pal/job_application_tracker.csv"
CLOSED_PHRASES = [
    "job not found",
    "no longer available",
    "position has been filled",
    "role is closed",
    "this job has expired",
    "not accepting applications",
    "requisition is closed",
]
AGGREGATORS = {"indeed.com", "ziprecruiter.com", "glassdoor.com", "linkedin.com", "monster.com"}
ATS_HOSTS = {"greenhouse.io", "lever.co", "myworkdayjobs.com", "ashbyhq.com", "smartrecruiters.com", "icims.com"}
REQUIRED_TRACKER_COLUMNS = [
    "App #",
    "Company",
    "Position",
    "Job ID",
    "Location",
    "Pay",
    "ATS System",
    "Application URL",
    "Resume File",
    "Cover Letter File",
    "Date Applied",
    "Status",
]
VERIFICATION_BASE_COLUMNS = [
    "Company",
    "Position",
    "Work Arrangement",
    "URL",
    "Status",
    "Reason",
    "Date Added",
    "Source",
    "Folder Path",
]

_TEST_TEMP_ROOT: Optional[str] = None


def normalize_hostname(host: str) -> str:
    host = (host or "").lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def is_blocked_host(host: str) -> bool:
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    blocked = [
        ipaddress.ip_network("127.0.0.0/8"),
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("169.254.0.0/16"),
        ipaddress.ip_network("0.0.0.0/8"),
        ipaddress.ip_network("100.64.0.0/10"),
        ipaddress.ip_network("::1/128"),
        ipaddress.ip_network("fc00::/7"),
        ipaddress.ip_network("fe80::/10"),
    ]
    return any(ip in net for net in blocked)


def is_url_safe(url: str) -> bool:
    """>>> is_url_safe("http://127.0.0.1")
False
>>> is_url_safe("https://93.184.216.34")
True
"""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = parsed.hostname
    if not host:
        return False
    if is_blocked_host(host):
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        resolved = info[4][0]
        if is_blocked_host(resolved):
            return False
    return True


def ghost_closed_detect(text: str) -> bool:
    """>>> ghost_closed_detect("This position has been filled")
True
>>> ghost_closed_detect("Great opportunity available")
False
"""
    lowered = text.lower()
    return any(ph in lowered for ph in CLOSED_PHRASES)


def csv_roundtrip_demo() -> bool:
    """>>> csv_roundtrip_demo()
True
"""
    global _TEST_TEMP_ROOT
    with tempfile.TemporaryDirectory() as tmp:
        prev_env = os.environ.get("GATEKEEPER_TEST_MODE")
        os.environ["GATEKEEPER_TEST_MODE"] = "1"
        _TEST_TEMP_ROOT = tmp
        path = os.path.join(tmp, "roundtrip.csv")
        rows = [{"a": "1", "b": "2"}]
        write_csv_atomic(path, rows, ["a", "b"], None, allow_test=True)
        read = read_csv(path)
        if prev_env is None:
            os.environ.pop("GATEKEEPER_TEST_MODE", None)
        else:
            os.environ["GATEKEEPER_TEST_MODE"] = prev_env
        _TEST_TEMP_ROOT = None
        return read == rows


def setup_logging(quiet: bool, verbose: bool) -> None:
    level = logging.INFO
    if quiet:
        level = logging.WARNING
    if verbose:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def is_within(path: str, root: str) -> bool:
    real = os.path.realpath(path)
    root_real = os.path.realpath(root)
    return real == root_real or real.startswith(root_real + os.sep)


def _allow_test_writes(path: str) -> bool:
    if os.environ.get("GATEKEEPER_TEST_MODE") != "1":
        return False
    if not _TEST_TEMP_ROOT:
        return False
    return is_within(path, _TEST_TEMP_ROOT)


def safe_file_write(path: str, args) -> bool:
    real = os.path.realpath(path)
    pal_root = os.path.realpath(args.pal_dir)
    js_root = os.path.join(pal_root, "Job_Search_System")
    exclusions_expected = os.path.realpath(os.path.join(js_root, "search_exclusions.csv"))
    verification_expected = os.path.realpath(os.path.join(js_root, "internship_verification_list.csv"))
    tracker_expected = os.path.realpath(os.path.join(pal_root, "job_application_tracker.csv"))
    exclusions_ok = real == exclusions_expected and real == os.path.realpath(args.exclusions)
    verification_ok = real == verification_expected and real == os.path.realpath(args.verification)
    tracker_ok = real == tracker_expected and real == os.path.realpath(args.tracker)
    return exclusions_ok or verification_ok or tracker_ok


def is_application_folder(path: str, allowed_roots: List[str]) -> bool:
    real = os.path.realpath(path)
    if not os.path.basename(real).endswith("_Application"):
        return False
    return any(is_within(real, root) for root in allowed_roots)


def triage_allowed_roots(args) -> List[str]:
    return [os.path.realpath(args.need_to_apply), os.path.realpath(os.path.join(args.pal_dir, "Need_To_Verify"))]


def safe_folder_action(path: str, args, allow_pal_root: bool = False) -> bool:
    roots = triage_allowed_roots(args)
    if allow_pal_root:
        roots.append(os.path.realpath(args.pal_dir))
    return is_application_folder(path, roots)


def read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{k: v for k, v in row.items()} for row in reader]


def write_csv_atomic(path: str, rows: List[Dict[str, str]], fieldnames: List[str], args, allow_test: bool = False) -> None:
    if args is not None:
        if not safe_file_write(path, args):
            raise ValueError("Unsafe write path")
    elif not (allow_test and _allow_test_writes(path)):
        raise ValueError("Unsafe write path")
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_gatekeeper_", text=True)
    os.close(fd)
    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def merge_fieldnames(existing: List[str], required: List[str]) -> List[str]:
    present = list(existing) if existing else []
    for col in required:
        if col not in present:
            present.append(col)
    return present


def load_or_init_csv(
    path: str, base_header: List[str], args, write_allowed: bool
) -> Tuple[List[Dict[str, str]], List[str], bool]:
    if not os.path.exists(path):
        if write_allowed:
            write_csv_atomic(path, [], base_header, args)
        return [], base_header, True
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or base_header
        rows = [{k: v for k, v in row.items()} for row in reader]
    header = merge_fieldnames(header, base_header)
    return rows, header, False


def extract_url_from_json(data: Dict[str, str]) -> str:
    for key in ["application_url", "url", "job_url", "posting_url", "apply_url", "link"]:
        if key in data and data[key]:
            return str(data[key])
    return ""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch_url(url: str, timeout: int, max_bytes: int, user_agent: str) -> Tuple[int, bytes, str]:
    if not is_url_safe(url):
        raise ValueError("Unsafe URL")
    opener = urllib.request.build_opener(NoRedirect())
    opener.addheaders = [("User-Agent", user_agent)]
    current = url
    seen = set()
    for _ in range(5):
        if current in seen:
            raise ValueError("Redirect loop")
        seen.add(current)
        req = urllib.request.Request(current)
        try:
            with opener.open(req, timeout=timeout) as resp:
                status = resp.getcode()
                data = resp.read(max_bytes + 1)
                content = data[:max_bytes]
                if not is_url_safe(resp.geturl()):
                    raise ValueError("Unsafe final URL")
                if status in (301, 302, 303, 307, 308) and resp.getheader("Location"):
                    location = urllib.parse.urljoin(resp.geturl(), resp.getheader("Location"))
                    if not is_url_safe(location):
                        raise ValueError("Unsafe redirect")
                    current = location
                    continue
                return status, content, resp.geturl()
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                location = e.headers.get("Location")
                if not location:
                    return e.code, b"", current
                location = urllib.parse.urljoin(current, location)
                if not is_url_safe(location):
                    raise ValueError("Unsafe redirect")
                current = location
                continue
            raise
    raise ValueError("Too many redirects")


def host_matches_ats(host: str) -> bool:
    norm = normalize_hostname(host)
    for candidate in ATS_HOSTS:
        cand_norm = normalize_hostname(candidate)
        if norm == cand_norm or norm.endswith("." + cand_norm):
            return True
    return False


def detect_canonical_links(content: str, base_url: str) -> Optional[str]:
    links = re.findall(r"href\s*=\s*['\"]([^'\"]+)['\"]", content, flags=re.IGNORECASE)
    for link in links:
        resolved = urllib.parse.urljoin(base_url, link)
        host = urllib.parse.urlparse(resolved).hostname or ""
        if host_matches_ats(host):
            return resolved
    return None


def classify_listing(status_code: int, content: str, final_url: str) -> Tuple[str, str]:
    lowered = content.lower()
    if status_code in (404, 410) or ghost_closed_detect(lowered):
        return "EXCLUDE", "Position closed"
    parsed = urllib.parse.urlparse(final_url)
    host = normalize_hostname(parsed.hostname or "") if parsed.hostname else ""
    canonical = detect_canonical_links(content, final_url)
    if any(host == normalize_hostname(agg) or host.endswith("." + normalize_hostname(agg)) for agg in AGGREGATORS):
        if not canonical:
            return "MONITOR", "Aggregator no ATS canonical"
        host = normalize_hostname(urllib.parse.urlparse(canonical).hostname or host)
        return "APPLY", f"Aggregator routed to ATS {host}"
    freshness = re.search(r"posted[^\d]*(\d+)\s+day", lowered)
    if freshness and int(freshness.group(1)) > 14:
        return "MONITOR", "Stale posting"
    if canonical:
        return "APPLY", "ATS canonical present"
    return "MONITOR", "No date evidence"


def diff_preview(changes: List[Tuple[Dict[str, str], Dict[str, str]]]) -> List[str]:
    lines = []
    for before, after in changes[:10]:
        b_status = before.get("Status", "")
        a_status = after.get("Status", "")
        b_reason = before.get("Reason", "")
        a_reason = after.get("Reason", "")
        lines.append(f"{before.get('Company','')} : {b_status}->{a_status} | {b_reason}->{a_reason}")
    return lines


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ingest(args) -> int:
    if not os.path.isdir(args.need_to_apply):
        sys.stderr.write(f"Need_To_Apply directory not found: {args.need_to_apply}\n")
        return 2
    rows, header, missing = load_or_init_csv(
        args.verification, VERIFICATION_BASE_COLUMNS, args, write_allowed=args.yes
    )
    existing_by_company = {r.get("Company", ""): r for r in rows}
    updated_rows = list(rows)
    added = []
    changed = []
    for entry in os.listdir(args.need_to_apply):
        if not entry.endswith("_Application"):
            continue
        folder = os.path.join(args.need_to_apply, entry)
        if not os.path.isdir(folder):
            continue
        company = entry.replace("_Application", "")
        jd_path = os.path.join(folder, "job_description.json")
        url = ""
        position = ""
        if os.path.exists(jd_path):
            try:
                with open(jd_path, encoding="utf-8") as f:
                    data = json.load(f)
                url = extract_url_from_json(data)
                position = str(data.get("position") or data.get("title") or "")
            except Exception as e:
                logging.warning("Failed to parse %s: %s", jd_path, e)
        if company in existing_by_company:
            row = dict(existing_by_company[company])
            row["URL"] = row.get("URL") or url
            row["Position"] = row.get("Position") or position
            row["Folder Path"] = folder
            idx = updated_rows.index(existing_by_company[company])
            updated_rows[idx] = row
            if row != existing_by_company[company]:
                changed.append((existing_by_company[company], row))
        else:
            new_row = {k: "" for k in header}
            new_row.update(
                {
                    "Company": company,
                    "URL": url,
                    "Position": position,
                    "Status": "PENDING",
                    "Reason": "",
                    "Date Added": dt.date.today().isoformat(),
                    "Source": "ingest",
                    "Folder Path": folder,
                }
            )
            updated_rows.append(new_row)
            added.append(({}, new_row))
    if not args.yes:
        print(f"PLAN ingest: read {len(rows)} rows, would write {len(updated_rows)} rows")
        if missing:
            print(f"Would create {args.verification}")
        for line in diff_preview(changed + added):
            print(line)
        return 0
    write_csv_atomic(args.verification, updated_rows, header, args)
    print(f"Applied ingest: wrote {len(updated_rows)} rows")
    for line in diff_preview(changed + added):
        print(line)
    return 0


def verify(args) -> int:
    rows, header, missing = load_or_init_csv(
        args.verification, VERIFICATION_BASE_COLUMNS, args, write_allowed=args.yes
    )
    updates = []
    for row in rows:
        if row.get("Status") not in {"PENDING", "MONITOR"}:
            continue
        if row.get("URL", ""):
            updates.append(row)
    if not args.yes:
        print(f"PLAN verify: would evaluate {len(updates)} rows; 0 status changes")
        if missing:
            print(f"Verification file missing; would create {args.verification}")
        for url in [r.get("URL", "") for r in updates][:10]:
            print(f"Would fetch: {url}")
        print("PLAN: no network in plan mode; cannot classify. Use --yes to verify.")
        return 0
    changes = []
    for row in updates:
        new_row = dict(row)
        status = row.get("Status", "")
        reason = row.get("Reason", "")
        try:
            code, content, final_url = fetch_url(row.get("URL", ""), args.timeout, args.max_bytes, args.user_agent)
            new_status, action_reason = classify_listing(code, content.decode("utf-8", errors="ignore"), final_url)
        except Exception as e:
            new_status, action_reason = "MONITOR", f"fetch error: {e}"
        new_row["Status"] = new_status
        new_row["Reason"] = action_reason
        if (new_status != status) or (action_reason != reason):
            changes.append((row, new_row))
            row.update(new_row)
    write_csv_atomic(args.verification, rows, header, args)
    print(f"Applied verify: updated {len(changes)} rows")
    for line in diff_preview(changes):
        print(line)
    return 0


def append_exclusion(path: str, company: str, reason: str, source: str, args) -> None:
    rows, header, _ = load_or_init_csv(
        path, ["Company", "Reason", "Date Added", "Source"], args, write_allowed=args.yes
    )
    rows.append(
        {
            "Company": company,
            "Reason": reason,
            "Date Added": dt.date.today().isoformat(),
            "Source": source,
        }
    )
    write_csv_atomic(path, rows, header, args)


def triage(args) -> int:
    rows, header, missing = load_or_init_csv(
        args.verification, VERIFICATION_BASE_COLUMNS, args, write_allowed=args.yes
    )
    changes = []
    need_to_verify = os.path.join(args.pal_dir, "Need_To_Verify")
    actions = []
    allowed_roots = triage_allowed_roots(args)
    for row in rows:
        folder = row.get("Folder Path", "")
        if not folder:
            continue
        if not os.path.exists(folder):
            logging.warning("Folder missing, skipping triage: %s", folder)
            continue
        status = row.get("Status", "")
        if not is_application_folder(folder, allowed_roots):
            continue
        if status == "EXCLUDE":
            actions.append(("delete", folder, row.get("Company", ""), row))
        elif status in {"PENDING", "MONITOR"}:
            dest = os.path.join(need_to_verify, os.path.basename(folder))
            actions.append(("move", folder, dest, row, "to_verify"))
        elif status == "APPLY" and is_within(os.path.realpath(folder), os.path.realpath(need_to_verify)):
            dest = os.path.join(os.path.realpath(args.need_to_apply), os.path.basename(folder))
            actions.append(("move", folder, dest, row, "to_apply"))
    if not args.yes:
        print(f"PLAN triage: {len(actions)} actions")
        if missing:
            print(f"Verification file missing; would create {args.verification}")
        for act in actions[:10]:
            kind = act[0]
            if kind == "delete":
                _, src, company, _ = act
                print(f"DELETE {src} ({company})")
            else:
                _, src, tgt, row, _ = act
                print(f"MOVE {src} -> {tgt} ({row.get('Company','')})")
        return 0
    ensure_directory(need_to_verify)
    for act in actions:
        if act[0] == "delete":
            _, src, _, row = act
            if is_application_folder(src, allowed_roots) and os.path.exists(src):
                shutil.rmtree(src, ignore_errors=True)
                append_exclusion(args.exclusions, row.get("Company", ""), "Verification exclude", "triage", args)
        else:
            _, src, tgt, row, _ = act
            src_ok = is_application_folder(src, allowed_roots)
            tgt_ok = is_application_folder(tgt, [os.path.realpath(args.need_to_apply), os.path.realpath(need_to_verify)])
            if src_ok and tgt_ok:
                ensure_directory(os.path.dirname(tgt))
                before = dict(row)
                shutil.move(src, tgt)
                row["Folder Path"] = tgt
                changes.append((before, dict(row)))
    write_csv_atomic(args.verification, rows, header, args)
    print(f"Applied triage: {len(actions)} actions; {len(changes)} row updates")
    for line in diff_preview(changes):
        print(line)
    for act in actions[:10]:
        if act[0] == "delete":
            print(f"Deleted {act[1]}")
        else:
            print(f"Moved {act[1]} -> {act[2]}")
    return 0


def next_app_number(rows: List[Dict[str, str]]) -> int:
    nums = []
    for row in rows:
        try:
            nums.append(int(row.get("App #", "") or 0))
        except ValueError:
            continue
    return (max(nums) if nums else 0) + 1


def load_job_description(company_folder: str) -> Dict[str, str]:
    jd_path = os.path.join(company_folder, "job_description.json")
    if not os.path.exists(jd_path):
        return {}
    try:
        with open(jd_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def mark_applied(args) -> int:
    tracker_rows, tracker_header, tracker_missing = load_or_init_csv(
        args.tracker, REQUIRED_TRACKER_COLUMNS, args, write_allowed=args.yes
    )
    exclusions_rows, exclusions_header, exclusions_missing = load_or_init_csv(
        args.exclusions, ["Company", "Reason", "Date Added", "Source"], args, write_allowed=args.yes
    )
    company_folder = os.path.join(args.need_to_apply, f"{args.company}_Application")
    dest_folder = os.path.join(args.pal_dir, f"{args.company}_Application")
    jd = load_job_description(company_folder)
    position = jd.get("position") or jd.get("title") or ""
    application_url = extract_url_from_json(jd) if jd else ""
    date_applied = args.date or dt.date.today().isoformat()
    app_num = next_app_number(tracker_rows)
    tracker_header = merge_fieldnames(tracker_header, REQUIRED_TRACKER_COLUMNS)
    new_tracker_row = {k: "" for k in tracker_header}
    new_tracker_row.update(
        {
            "App #": str(app_num),
            "Company": args.company,
            "Position": position,
            "Application URL": application_url,
            "Date Applied": date_applied,
            "Status": args.status,
        }
    )
    new_exclusion_row = {
        "Company": args.company,
        "Reason": "Applied",
        "Date Added": date_applied,
        "Source": "application_submitted",
    }
    if not os.path.exists(company_folder):
        print("Application folder missing; cannot mark applied")
        return 2
    if os.path.exists(dest_folder):
        print("Destination already exists; aborting to avoid overwrite")
        return 3
    if not args.yes:
        print("PLAN mark-applied:")
        if tracker_missing:
            print(f"Would create {args.tracker}")
        if exclusions_missing:
            print(f"Would create {args.exclusions}")
        print(f"Move {company_folder} -> {dest_folder}")
        print(f"Append exclusion: {new_exclusion_row}")
        print(f"Append tracker row: {new_tracker_row}")
        return 0
    if not safe_folder_action(company_folder, args, allow_pal_root=True) or not is_application_folder(
        dest_folder, [os.path.realpath(args.pal_dir)]
    ):
        print("Unsafe path")
        return 1
    shutil.move(company_folder, dest_folder)
    exclusions_rows.append(new_exclusion_row)
    write_csv_atomic(args.exclusions, exclusions_rows, exclusions_header, args)
    tracker_rows.append(new_tracker_row)
    write_csv_atomic(args.tracker, tracker_rows, tracker_header, args)
    print("Applied mark-applied")
    return 0


def exclude(args) -> int:
    rows, header, missing = load_or_init_csv(
        args.verification, VERIFICATION_BASE_COLUMNS, args, write_allowed=args.yes
    )
    target_folder = os.path.join(args.need_to_apply, f"{args.company}_Application")
    changed = []
    found = False
    for row in rows:
        if row.get("Company") == args.company:
            new_row = dict(row)
            new_row["Status"] = "EXCLUDE"
            new_row["Reason"] = args.reason
            changed.append((row, new_row))
            row.update(new_row)
            found = True
            break
    if not found:
        new_row = {k: "" for k in header}
        new_row.update(
            {
                "Company": args.company,
                "Status": "EXCLUDE",
                "Reason": args.reason,
                "Date Added": dt.date.today().isoformat(),
                "Source": "exclude_command",
                "Folder Path": target_folder if os.path.exists(target_folder) else "",
            }
        )
        rows.append(new_row)
        changed.append(({}, new_row))
    if not args.yes:
        print("PLAN exclude:")
        if missing:
            print(f"Verification file missing; would create {args.verification}")
        print(f"Delete {target_folder}")
        for line in diff_preview(changed):
            print(line)
        return 0
    if safe_folder_action(target_folder, args) and os.path.exists(target_folder):
        shutil.rmtree(target_folder, ignore_errors=True)
    append_exclusion(args.exclusions, args.company, args.reason, "exclude_command", args)
    write_csv_atomic(args.verification, rows, header, args)
    print("Applied exclude")
    for line in diff_preview(changed):
        print(line)
    return 0


def self_test(args) -> int:
    global _TEST_TEMP_ROOT
    prev_env = os.environ.get("GATEKEEPER_TEST_MODE")
    doctest_failures = 0
    plan_ok = True
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["GATEKEEPER_TEST_MODE"] = "1"
            _TEST_TEMP_ROOT = tmp
            doctest_failures, _ = doctest.testmod()
            temp_pal = tmp
            temp_need = os.path.join(tmp, "Need_To_Apply")
            os.makedirs(temp_need, exist_ok=True)
            temp_js = os.path.join(temp_pal, "Job_Search_System")
            os.makedirs(temp_js, exist_ok=True)
            temp_ver = os.path.join(temp_js, "internship_verification_list.csv")
            temp_exc = os.path.join(temp_js, "search_exclusions.csv")
            temp_tracker = os.path.join(temp_pal, "job_application_tracker.csv")
            ingest_args = argparse.Namespace(
                pal_dir=temp_pal,
                need_to_apply=temp_need,
                exclusions=temp_exc,
                verification=temp_ver,
                tracker=temp_tracker,
                yes=False,
                verbose=False,
                quiet=False,
                timeout=args.timeout,
                max_bytes=args.max_bytes,
                user_agent=args.user_agent,
                command="ingest",
            )
            verify_args = argparse.Namespace(
                pal_dir=temp_pal,
                need_to_apply=temp_need,
                exclusions=temp_exc,
                verification=temp_ver,
                tracker=temp_tracker,
                yes=False,
                verbose=False,
                quiet=False,
                timeout=args.timeout,
                max_bytes=args.max_bytes,
                user_agent=args.user_agent,
                command="verify",
            )
            if ingest(ingest_args) != 0:
                plan_ok = False
            if verify(verify_args) != 0:
                plan_ok = False
    finally:
        if prev_env is None:
            os.environ.pop("GATEKEEPER_TEST_MODE", None)
        else:
            os.environ["GATEKEEPER_TEST_MODE"] = prev_env
        _TEST_TEMP_ROOT = None
    if doctest_failures == 0 and plan_ok:
        print("self-test passed")
        return 0
    print(f"self-test failed: doctest_failures={doctest_failures}, plan_ok={plan_ok}")
    return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Internship pipeline gatekeeper")
    parser.add_argument("--pal-dir", default=PAL_ROOT_DEFAULT)
    parser.add_argument("--need-to-apply", default=NEED_TO_APPLY_DEFAULT)
    parser.add_argument("--exclusions", default=EXCLUSIONS_DEFAULT)
    parser.add_argument("--verification", default=VERIFICATION_DEFAULT)
    parser.add_argument("--tracker", default=TRACKER_DEFAULT)
    parser.add_argument("--yes", action="store_true", help="Apply changes; otherwise plan-only")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--max-bytes", type=int, default=1_500_000)
    parser.add_argument("--user-agent", default="Mozilla/5.0 (Gatekeeper)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Ingest folders into verification list")
    subparsers.add_parser("verify", help="Verify pending/monitor listings")
    subparsers.add_parser("triage", help="Move folders based on verification status")
    subparsers.add_parser("self-test", help="Run internal self-tests")
    mark = subparsers.add_parser("mark-applied", help="Mark application as applied")
    mark.add_argument("--company", required=True)
    mark.add_argument("--status", default="Applied")
    mark.add_argument("--date", default=None)
    exc = subparsers.add_parser("exclude", help="Exclude a company")
    exc.add_argument("--company", required=True)
    exc.add_argument("--reason", default="Not Interested")
    args = parser.parse_args(argv)
    setup_logging(args.quiet, args.verbose)
    if not args.yes and args.command == "verify":
        logging.info("Plan mode: no network requests will be made")
    if args.command == "ingest":
        return ingest(args)
    if args.command == "verify":
        return verify(args)
    if args.command == "triage":
        return triage(args)
    if args.command == "mark-applied":
        return mark_applied(args)
    if args.command == "exclude":
        return exclude(args)
    if args.command == "self-test":
        return self_test(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
