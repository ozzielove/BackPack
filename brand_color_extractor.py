"""Brand color extractor script.

Zero-trust palette extraction with caching, SSRF protections, and provenance-aware
weighting. Network and writes only occur with --yes (unless a cached palette is
served). Doctest examples cover key helpers.

Usage example:
    python brand_color_extractor.py https://example.com --output out.json --yes

>>> normalize_hex('#0f0')
'#00ff00'
>>> normalize_hex('rgb(255, 0, 0)')
'#ff0000'
>>> normalize_hex('rgba(0, 0, 255, 0.5)')
'#7f7fff'
>>> round(relative_luminance((255, 255, 255)), 3)
1.0
>>> round(relative_luminance((0, 0, 0)), 3)
0.0
>>> choose_text_color('#ffffff', '#000000')
'#000000'
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gzip
import io
import json
import logging
import os
import random
import re
import socket
import sys
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Dict, Iterable, List, Optional, Tuple

USER_AGENT = "BrandPaletteExtractor/1.1"
MAX_REDIRECTS = 5
CSS_COLOR_PROPS = {
    "color",
    "background",
    "background-color",
    "border-color",
    "fill",
    "stroke",
}
NAMED_COLORS = {
    "black": "#000000",
    "white": "#ffffff",
    "red": "#ff0000",
    "green": "#00ff00",
    "blue": "#0000ff",
}
PRIVATE_NETWORKS_V4 = [
    ("127.0.0.0", 8),
    ("10.0.0.0", 8),
    ("172.16.0.0", 12),
    ("192.168.0.0", 16),
    ("169.254.0.0", 16),
    ("0.0.0.0", 8),
    ("100.64.0.0", 10),
]
PRIVATE_NETWORKS_V6 = [
    ("::1", 128),
    ("fc00::", 7),
    ("fe80::", 10),
]
DEFAULT_MEMORY_DIR = os.path.expanduser("~/Downloads/Pal/.pipeline_memory/slide_decks/")
LOCK_TTL_SECONDS = 300


class LimitedHTMLParser(HTMLParser):
    """Collects metadata without executing scripts."""

    def __init__(self) -> None:
        super().__init__()
        self.meta_theme_color: List[str] = []
        self.meta_site_name: Optional[str] = None
        self.meta_app_name: Optional[str] = None
        self.title: Optional[str] = None
        self.in_title = False
        self.style_blocks: List[str] = []
        self.inline_styles: List[str] = []
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        if tag == "meta":
            name = attr_map.get("name", "").lower()
            prop = attr_map.get("property", "").lower()
            content = attr_map.get("content")
            if name == "theme-color" and content:
                self.meta_theme_color.append(content)
            if prop == "og:site_name" and content:
                self.meta_site_name = content
            if name == "application-name" and content:
                self.meta_app_name = content
        if tag == "title":
            self.in_title = True
        if tag == "style":
            self._current_style = []  # type: ignore[attr-defined]
        if tag == "link":
            rel = attr_map.get("rel", "").lower()
            href = attr_map.get("href")
            if rel and "stylesheet" in rel and href:
                self.links.append(href)
        style_attr = attr_map.get("style")
        if style_attr:
            self.inline_styles.append(style_attr)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        if tag == "style":
            if hasattr(self, "_current_style"):
                self.style_blocks.append("".join(getattr(self, "_current_style")))
                delattr(self, "_current_style")

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title = (self.title or "") + data
        if hasattr(self, "_current_style"):
            getattr(self, "_current_style").append(data)


def relative_luminance(rgb: Tuple[int, int, int]) -> float:
    """Calculate relative luminance per WCAG.

    >>> round(relative_luminance((255, 255, 255)), 3)
    1.0
    """

    def channel(c: int) -> float:
        v = c / 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def parse_hex_color(value: str) -> Optional[Tuple[int, int, int]]:
    value = value.strip().lower()
    if value in NAMED_COLORS:
        value = NAMED_COLORS[value]
    if re.fullmatch(r"#[0-9a-f]{3}", value):
        r, g, b = (int(value[i], 16) * 17 for i in range(1, 4))
        return (r, g, b)
    if re.fullmatch(r"#[0-9a-f]{6}", value):
        return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))
    return None


def parse_rgb(value: str) -> Optional[Tuple[int, int, int, float]]:
    m = re.fullmatch(r"rgba?\(\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})(?:\s*,\s*([0-9.]+))?\s*\)", value.strip().lower())
    if not m:
        return None
    r, g, b = (max(0, min(255, int(m.group(i)))) for i in range(1, 4))
    a = float(m.group(4)) if m.group(4) is not None else 1.0
    return (r, g, b, max(0.0, min(1.0, a)))


def parse_hsl(value: str) -> Optional[Tuple[int, int, int, float]]:
    m = re.fullmatch(r"hsla?\(\s*([-0-9.]+)\s*,\s*([-0-9.]+)%\s*,\s*([-0-9.]+)%(?:\s*,\s*([0-9.]+))?\s*\)", value.strip().lower())
    if not m:
        return None
    h = float(m.group(1)) % 360
    s = max(0.0, min(100.0, float(m.group(2)))) / 100.0
    l = max(0.0, min(100.0, float(m.group(3)))) / 100.0
    a = float(m.group(4)) if m.group(4) is not None else 1.0

    def hue_to_rgb(p: float, q: float, t: float) -> float:
        t = t % 1
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 1 / 2:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p

    if s == 0:
        r = g = b = int(l * 255)
    else:
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = int(round(hue_to_rgb(p, q, h / 360 + 1 / 3) * 255))
        g = int(round(hue_to_rgb(p, q, h / 360) * 255))
        b = int(round(hue_to_rgb(p, q, h / 360 - 1 / 3) * 255))
    return (
        max(0, min(255, r)),
        max(0, min(255, g)),
        max(0, min(255, b)),
        max(0.0, min(1.0, a)),
    )


def normalize_hex(value: str) -> Optional[str]:
    """Convert supported color formats to #RRGGBB.

    >>> normalize_hex('#abc')
    '#aabbcc'
    >>> normalize_hex('hsl(0, 100%, 50%)')
    '#ff0000'
    """

    value = value.strip()
    if not value:
        return None
    hex_rgb = parse_hex_color(value)
    if hex_rgb:
        return "#%02x%02x%02x" % hex_rgb
    rgb = parse_rgb(value)
    if rgb:
        r, g, b, a = rgb
        r = int(round(r * a + 255 * (1 - a)))
        g = int(round(g * a + 255 * (1 - a)))
        b = int(round(b * a + 255 * (1 - a)))
        return "#%02x%02x%02x" % (r, g, b)
    hsl = parse_hsl(value)
    if hsl:
        r, g, b, a = hsl
        r = int(round(r * a + 255 * (1 - a)))
        g = int(round(g * a + 255 * (1 - a)))
        b = int(round(b * a + 255 * (1 - a)))
        return "#%02x%02x%02x" % (r, g, b)
    return None


def choose_text_color(bg_hex: str, preferred: str) -> str:
    bg = parse_hex_color(bg_hex) or (255, 255, 255)
    preferred_rgb = parse_hex_color(preferred) or (0, 0, 0)
    black = (0, 0, 0)
    white = (255, 255, 255)
    def contrast(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
        l1 = relative_luminance(c1) + 0.05
        l2 = relative_luminance(c2) + 0.05
        return max(l1, l2) / min(l1, l2)
    best = preferred_rgb
    if contrast(bg, preferred_rgb) < 4.5:
        best = black if contrast(bg, black) >= contrast(bg, white) else white
    return "#%02x%02x%02x" % best


def is_private_ip(hostname: str) -> bool:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except OSError:
        return True
    for family, _, _, _, sockaddr in infos:
        ip = sockaddr[0]
        if family == socket.AF_INET:
            if any(_in_cidr(ip, net, bits) for net, bits in PRIVATE_NETWORKS_V4):
                return True
        elif family == socket.AF_INET6:
            if any(_in_cidr_v6(ip, net, bits) for net, bits in PRIVATE_NETWORKS_V6):
                return True
    return False


def _in_cidr(ip: str, network: str, bits: int) -> bool:
    import struct
    import ipaddress

    mask = (0xFFFFFFFF << (32 - bits)) & 0xFFFFFFFF
    ip_int = int(ipaddress.IPv4Address(ip))
    net_int = int(ipaddress.IPv4Address(network))
    return ip_int & mask == net_int & mask


def _in_cidr_v6(ip: str, network: str, bits: int) -> bool:
    import ipaddress

    ip_int = int(ipaddress.IPv6Address(ip))
    net_int = int(ipaddress.IPv6Address(network))
    mask = (1 << 128) - (1 << (128 - bits))
    return ip_int & mask == net_int & mask


@dataclasses.dataclass
class MemoryRecord:
    company: str
    url: str
    domain: str
    palette: Dict[str, Optional[str]]
    theme: str
    confidence: float
    sources_used: List[str]
    extracted_at: str
    pipeline_version: str
    memory_state_id: str


class FileLock:
    def __init__(self, path: str, ttl: int = LOCK_TTL_SECONDS) -> None:
        self.path = path
        self.ttl = ttl

    def __enter__(self):
        start = time.time()
        while True:
            if not os.path.exists(self.path):
                try:
                    data = {"pid": os.getpid(), "created_at": _utcnow()}
                    tmp = self.path + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(data, f)
                    os.replace(tmp, self.path)
                    return self
                except OSError:
                    pass
            else:
                try:
                    with open(self.path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    created = meta.get("created_at")
                    if created and _is_stale(created, self.ttl):
                        os.remove(self.path)
                        continue
                except Exception:
                    try:
                        os.remove(self.path)
                        continue
                    except OSError:
                        pass
            time.sleep(0.05)
            if time.time() - start > self.ttl:
                raise RuntimeError("Lock wait exceeded TTL")

    def __exit__(self, exc_type, exc, tb):
        with contextlib.suppress(OSError):
            os.remove(self.path)


def _is_stale(created_at: str, ttl: int) -> bool:
    try:
        ts = time.mktime(time.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ"))
    except Exception:
        return True
    return (time.time() - ts) > ttl


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_json(path: str, data: object) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def fetch_url(url: str, timeout: int, max_bytes: int, allow_redirects: int = MAX_REDIRECTS) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https allowed")
    if not parsed.hostname:
        raise ValueError("Invalid URL: missing host")
    if is_private_ip(parsed.hostname):
        raise ValueError("Refusing to fetch private or link-local host")

    seen = 0
    current = url
    while True:
        if seen > allow_redirects:
            raise ValueError("Too many redirects")
        req = urllib.request.Request(current, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            if status in (301, 302, 303, 307, 308):
                target = resp.getheader("Location")
                if not target:
                    raise ValueError("Redirect without Location header")
                current = urllib.parse.urljoin(current, target)
                parsed = urllib.parse.urlparse(current)
                if parsed.scheme not in {"http", "https"}:
                    raise ValueError("Redirected to non-http/https")
                if not parsed.hostname or is_private_ip(parsed.hostname):
                    raise ValueError("Redirected to private or invalid host")
                seen += 1
                continue
            raw = resp.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raise ValueError("Response exceeds byte limit")
            encoding = resp.getheader("Content-Encoding", "").lower()
            if encoding == "gzip":
                with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                    raw = gz.read(max_bytes + 1)
                    if len(raw) > max_bytes:
                        raise ValueError("Decompressed response too large")
            return raw


def parse_html(content: bytes, encoding: str = "utf-8") -> LimitedHTMLParser:
    parser = LimitedHTMLParser()
    parser.feed(content.decode(encoding, errors="ignore"))
    return parser


def collect_colors(parser: LimitedHTMLParser) -> Dict[str, List[Tuple[str, int]]]:
    colors: Dict[str, List[Tuple[str, int]]] = {}
    for val in parser.meta_theme_color:
        hex_val = normalize_hex(val)
        if hex_val:
            colors.setdefault(hex_val, []).append(("meta:theme-color", 6))
    for block in parser.style_blocks:
        for name, val in extract_css_vars(block):
            hex_val = normalize_hex(val)
            if hex_val:
                weight = 5 if re.search(r"(primary|brand|main)", name, re.I) else 3 if re.search(r"(accent|secondary|highlight|bg|background|text|fg|foreground)", name, re.I) else 1
                colors.setdefault(hex_val, []).append((f"css_var:{name}", weight))
        for prop, val in extract_css_props(block):
            hex_val = normalize_hex(val)
            if hex_val:
                colors.setdefault(hex_val, []).append((f"css_prop:{prop}", 1))
    for inline in parser.inline_styles:
        for prop, val in extract_inline_style(inline):
            hex_val = normalize_hex(val)
            if hex_val:
                colors.setdefault(hex_val, []).append((f"inline:{prop}", 2))
    return colors


def extract_css_vars(css: str) -> List[Tuple[str, str]]:
    return re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", css)


def extract_css_props(css: str) -> List[Tuple[str, str]]:
    pattern = r"(" + "|".join(map(re.escape, CSS_COLOR_PROPS)) + r")\s*:\s*([^;]+);"
    return re.findall(pattern, css, flags=re.IGNORECASE)


def extract_inline_style(style: str) -> List[Tuple[str, str]]:
    parts = re.findall(r"([^:;]+):\s*([^;]+)", style)
    return [(k.strip().lower(), v.strip()) for k, v in parts if k.strip().lower() in CSS_COLOR_PROPS]


def fetch_stylesheets(base_url: str, links: List[str], args: argparse.Namespace) -> List[str]:
    fetched: List[str] = []
    base_host = urllib.parse.urlparse(base_url).hostname
    for href in links:
        if len(fetched) >= args.max_css_files:
            break
        resolved = urllib.parse.urljoin(base_url, href)
        parsed = urllib.parse.urlparse(resolved)
        host = parsed.hostname
        if not host:
            continue
        same_origin_allowed = args.same_origin_css
        cross_allowed = args.allow_css_cdn and args.css_host_allowlist and host in args.css_host_allowlist and not args.same_origin_css
        if not ((same_origin_allowed and host == base_host) or cross_allowed):
            continue
        if parsed.scheme not in {"http", "https"}:
            continue
        if is_private_ip(host):
            continue
        try:
            raw = fetch_url(resolved, args.timeout, args.max_css_bytes)
        except Exception as exc:  # noqa: BLE001
            logging.debug("Skip stylesheet %s: %s", resolved, exc)
            continue
        fetched.append(raw.decode("utf-8", errors="ignore"))
    return fetched


def weight_colors(colors: Dict[str, List[Tuple[str, int]]], external: Dict[str, List[Tuple[str, int]]]) -> Dict[str, List[Tuple[str, int]]]:
    merged: Dict[str, List[Tuple[str, int]]] = {k: v[:] for k, v in colors.items()}
    for hex_val, provs in external.items():
        merged.setdefault(hex_val, []).extend(provs)
    return merged


def select_palette(weighted: Dict[str, List[Tuple[str, int]]]) -> Tuple[Dict[str, Optional[str]], float, List[str]]:
    scores: Dict[str, int] = {c: sum(w for _, w in provs) for c, provs in weighted.items()}
    sorted_colors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    def prov_strings(hex_val: str) -> List[str]:
        return [p for p, _ in weighted.get(hex_val, [])]

    def is_neutral(hex_val: str) -> bool:
        lum = relative_luminance(parse_hex_color(hex_val) or (0, 0, 0))
        if lum > 0.95 or lum < 0.05:
            tags = " ".join(prov_strings(hex_val)).lower()
            if re.search(r"(text|fg|foreground|bg|background)", tags):
                return False
            return True
        return False

    chosen: Dict[str, Optional[str]] = {"primary": None, "secondary": None, "background": None, "text": None, "accent": None}
    non_neutral = [c for c, _ in sorted_colors if not is_neutral(c)]
    neutral = [c for c, _ in sorted_colors if is_neutral(c)]
    if non_neutral:
        chosen["primary"] = non_neutral[0]
    def distinct(existing: str, candidate: str) -> bool:
        a = parse_hex_color(existing)
        b = parse_hex_color(candidate)
        if not a or not b:
            return False
        return sum(abs(x - y) for x, y in zip(a, b)) >= 40
    for c in non_neutral[1:]:
        if chosen["primary"] and distinct(chosen["primary"], c):
            chosen["secondary"] = c
            break
    if chosen["secondary"] is None:
        for c in neutral:
            if chosen["primary"] and distinct(chosen["primary"], c):
                chosen["secondary"] = c
                break
    for c in non_neutral:
        if chosen["primary"] and chosen["secondary"] and distinct(chosen["primary"], c) and distinct(chosen["secondary"], c):
            chosen["accent"] = c
            break
    bg_candidates = [c for c in sorted_colors]
    bg = None
    for c, _ in bg_candidates:
        if re.search(r"(bg|background)", " ".join(prov_strings(c)), re.I):
            bg = c
            break
    if not bg:
        bg = neutral[0] if neutral else (chosen["secondary"] or chosen["primary"])
    chosen["background"] = bg
    text = None
    for c, _ in sorted_colors:
        if re.search(r"(text|fg|foreground)", " ".join(prov_strings(c)), re.I):
            text = c
            break
    if not text:
        text = choose_text_color(chosen["background"] or "#ffffff", "#000000")
    chosen["text"] = text

    sources_used = set()
    for provs in weighted.values():
        for p, _ in provs:
            if p.startswith("meta"):
                sources_used.add("meta")
            elif p.startswith("css_var") or p.startswith("css_prop"):
                sources_used.add("style_block")
            elif p.startswith("inline"):
                sources_used.add("inline")
            elif p.startswith("external_css"):
                sources_used.add("external_css")
    primary_provs = prov_strings(chosen["primary"] or "")
    confidence = 0.2
    if any(re.search(r"(primary|brand|main)", p, re.I) for p in primary_provs if p.startswith("css_var")):
        confidence += 0.3
    if any(p.startswith("meta:theme-color") for p in primary_provs):
        confidence += 0.2
    confidence += min(0.3, 0.1 * len(sources_used))
    secondary_score = scores.get(chosen.get("secondary") or "", 0)
    primary_score = scores.get(chosen.get("primary") or "", 0)
    if secondary_score >= 0:
        confidence += 0.2 * min(1.0, (primary_score / (secondary_score + 1e-9)) / 3)
    confidence = max(0.0, min(1.0, confidence))
    return chosen, confidence, sorted(sources_used)


def resolve_company_name(parser: LimitedHTMLParser, domain: str) -> str:
    if parser.meta_site_name:
        return parser.meta_site_name.strip()
    if parser.meta_app_name:
        return parser.meta_app_name.strip()
    if parser.title:
        return parser.title.strip()
    if domain:
        label = domain.split(".")[0]
        return label.capitalize()
    return "Unknown"


def load_memory(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        return {"latest_by_domain": {}, "history": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"latest_by_domain": {}, "history": []}


def ensure_allowed_output(output: Optional[str], memory_dir: str) -> None:
    if output is None:
        return
    abs_out = os.path.abspath(output)
    allowed_roots = {os.path.abspath(os.getcwd()), os.path.abspath(memory_dir)}
    if not any(abs_out.startswith(root + os.sep) or abs_out == root for root in allowed_roots):
        raise SystemExit("Output path must be within current working directory or memory-dir")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Extract brand colors without rendering HTML")
    parser.add_argument("url")
    parser.add_argument("--output")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--max-html-bytes", type=int, default=2_000_000)
    parser.add_argument("--max-css-bytes", type=int, default=1_000_000)
    parser.add_argument("--max-css-files", type=int, default=5)
    parser.add_argument("--same-origin-css", dest="same_origin_css", action="store_true", default=True)
    parser.add_argument("--no-same-origin-css", dest="same_origin_css", action="store_false")
    parser.add_argument("--allow-css-cdn", action="store_true")
    parser.add_argument("--css-host-allowlist", type=lambda s: [h.strip() for h in s.split(",") if h.strip()], default=[])
    parser.add_argument("--memory-dir", default=DEFAULT_MEMORY_DIR)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Perform network fetch and persist results")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.ERROR if args.quiet else logging.INFO, format="%(levelname)s:%(message)s")

    ensure_allowed_output(args.output, args.memory_dir)

    parsed_url = urllib.parse.urlparse(args.url)
    if parsed_url.scheme not in {"http", "https"}:
        raise SystemExit("Only http/https URLs are allowed")
    domain = parsed_url.hostname or ""

    mem_path = os.path.join(args.memory_dir, "extracted_palettes_history.json")
    os.makedirs(args.memory_dir, exist_ok=True)
    memory = load_memory(mem_path)
    cached = memory.get("latest_by_domain", {}).get(domain)
    if cached and not args.refresh:
        print(json.dumps(cached, ensure_ascii=False, indent=2, sort_keys=True))
        return

    plan = {
        "action": "extract_palette",
        "url": args.url,
        "domain": domain,
        "will_fetch": True,
        "will_write_memory": bool(args.yes),
        "will_write_output": bool(args.output and args.yes),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    if not args.yes:
        return

    html_bytes = fetch_url(args.url, args.timeout, args.max_html_bytes)
    parser_obj = parse_html(html_bytes)
    base_colors = collect_colors(parser_obj)
    stylesheets = fetch_stylesheets(args.url, parser_obj.links, args)
    external_colors: Dict[str, List[Tuple[str, int]]] = {}
    for css in stylesheets:
        for name, val in extract_css_vars(css):
            hex_val = normalize_hex(val)
            if hex_val:
                weight = 5 if re.search(r"(primary|brand|main)", name, re.I) else 3 if re.search(r"(accent|secondary|highlight|bg|background|text|fg|foreground)", name, re.I) else 1
                external_colors.setdefault(hex_val, []).append((f"external_css:css_var:{name}", weight))
        for prop, val in extract_css_props(css):
            hex_val = normalize_hex(val)
            if hex_val:
                external_colors.setdefault(hex_val, []).append((f"external_css:{prop}", 1))

    weighted = weight_colors(base_colors, external_colors)
    palette, confidence, sources_used = select_palette(weighted)
    bg_rgb = parse_hex_color(palette["background"] or "#ffffff") or (255, 255, 255)
    theme = "dark" if relative_luminance(bg_rgb) < 0.35 else "light"
    record = {
        "company": resolve_company_name(parser_obj, domain),
        "url": args.url,
        "domain": domain,
        "palette": palette,
        "theme": theme,
        "confidence": round(confidence, 3),
        "sources_used": sources_used,
        "extracted_at": _utcnow(),
        "pipeline_version": "3.0",
        "memory_state_id": _uuid4(),
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)
    with FileLock(mem_path + ".lock"):
        memory = load_memory(mem_path)
        memory.setdefault("latest_by_domain", {})[domain] = record
        memory.setdefault("history", []).append(record)
        atomic_write_json(mem_path, memory)
    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))


def _uuid4() -> str:
    import uuid

    return str(uuid.uuid4())


if __name__ == "__main__":
    main()
