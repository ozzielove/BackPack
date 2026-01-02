"""Brand color extractor script.

Extracts brand palette heuristically from a website without rendering HTML.

Usage example:
    python brand_color_extractor.py https://example.com --output out.json --yes

Doctest examples cover core parsing helpers:

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
import sys
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Dict, Iterable, List, Optional, Tuple

USER_AGENT = "BrandPaletteExtractor/1.0"
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
        if t < 1/6:
            return p + (q - p) * 6 * t
        if t < 1/2:
            return q
        if t < 2/3:
            return p + (q - p) * (2/3 - t) * 6
        return p
    if s == 0:
        r = g = b = int(l * 255)
    else:
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = int(round(hue_to_rgb(p, q, h/360 + 1/3) * 255))
        g = int(round(hue_to_rgb(p, q, h/360) * 255))
        b = int(round(hue_to_rgb(p, q, h/360 - 1/3) * 255))
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)), max(0.0, min(1.0, a)))


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


def choose_text_color(bg_hex: str, current_text: str) -> str:
    """Choose black/white improving contrast.

    >>> choose_text_color('#ffffff', '#000000')
    '#000000'
    """
    bg_rgb = parse_hex_color(bg_hex) or (255, 255, 255)
    txt_rgb = parse_hex_color(current_text) or (0, 0, 0)
    black = relative_luminance((0, 0, 0))
    white = relative_luminance((255, 255, 255))
    bg_l = relative_luminance(bg_rgb)
    def contrast(l1: float, l2: float) -> float:
        a, b = max(l1, l2), min(l1, l2)
        return (a + 0.05) / (b + 0.05)
    black_contrast = contrast(bg_l, black)
    white_contrast = contrast(bg_l, white)
    current_contrast = contrast(bg_l, relative_luminance(txt_rgb))
    if black_contrast >= white_contrast and black_contrast > current_contrast:
        return "#000000"
    if white_contrast > black_contrast and white_contrast > current_contrast:
        return "#ffffff"
    return "#%02x%02x%02x" % txt_rgb


def domain_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return parsed.hostname or parsed.netloc


def fetch_url(url: str, timeout: float, max_bytes: int) -> bytes:
    """Fetch a URL with redirect, byte, and gzip safeguards."""

    redirects = 0
    current = url
    while True:
        req = urllib.request.Request(current, headers={"User-Agent": USER_AGENT})
        with contextlib.closing(urllib.request.urlopen(req, timeout=timeout)) as resp:
            if resp.geturl() != current and redirects >= MAX_REDIRECTS:
                raise RuntimeError("Too many redirects")
            if resp.geturl() != current:
                redirects += 1
                current = resp.geturl()
                continue
            data = []
            total = 0
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise RuntimeError(f"Response exceeds limit {max_bytes} bytes")
                data.append(chunk)
            raw = b"".join(data)
            encoding = resp.headers.get("Content-Encoding", "").lower()
            if encoding == "gzip":
                with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                    decompressed = gz.read(max_bytes + 1)
                    if len(decompressed) > max_bytes:
                        raise RuntimeError(f"Decompressed response exceeds limit {max_bytes} bytes")
                    return decompressed
            return raw


def parse_colors_from_css(css: str) -> List[Tuple[str, int]]:
    results: List[Tuple[str, int]] = []
    for var, value in re.findall(r"(--[\w-]+)\s*:\s*([^;]+)", css):
        hex_value = normalize_hex(value.split()[0])
        if hex_value:
            weight = 5 if re.search(r"(primary|brand|main)", var, re.I) else 3 if re.search(r"(accent|secondary|highlight|bg|background|text|fg|foreground)", var, re.I) else 1
            results.append((hex_value, weight))
    for prop, value in re.findall(r"(color|background(?:-color)?|border-color|fill|stroke)\s*:\s*([^;]+)", css, flags=re.I):
        hex_value = normalize_hex(value.split()[0])
        if hex_value:
            results.append((hex_value, 1))
    return results


def parse_inline_styles(styles: Iterable[str]) -> List[Tuple[str, int]]:
    results: List[Tuple[str, int]] = []
    for style in styles:
        for prop, value in re.findall(r"(color|background(?:-color)?|border-color|fill|stroke)\s*:\s*([^;]+)", style, flags=re.I):
            hex_value = normalize_hex(value.split()[0])
            if hex_value:
                results.append((hex_value, 2))
    return results


def collect_sources(html: str) -> LimitedHTMLParser:
    parser = LimitedHTMLParser()
    parser.feed(html)
    return parser


def luminance_distance(hex1: str, hex2: str) -> float:
    rgb1 = parse_hex_color(hex1) or (0, 0, 0)
    rgb2 = parse_hex_color(hex2) or (0, 0, 0)
    return sum(abs(a - b) for a, b in zip(rgb1, rgb2))


def is_neutral(hex_value: str) -> bool:
    lum = relative_luminance(parse_hex_color(hex_value) or (0, 0, 0))
    return lum > 0.95 or lum < 0.05


def select_palette(weighted_colors: List[Tuple[str, int, str]]) -> Tuple[Dict[str, Optional[str]], str]:
    scores: Dict[str, int] = {}
    provenance: Dict[str, str] = {}
    for color, weight, source in weighted_colors:
        scores[color] = scores.get(color, 0) + weight
        provenance.setdefault(color, source)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    primary = None
    secondary = None
    accent = None
    background = None
    text = None
    for color, _ in ordered:
        if not primary and not is_neutral(color):
            primary = color
        elif primary and not secondary and not is_neutral(color) and luminance_distance(primary, color) >= 40:
            secondary = color
        elif primary and secondary and not accent and not is_neutral(color) and luminance_distance(primary, color) >= 40 and luminance_distance(secondary, color) >= 40:
            accent = color
    bg_candidates = [c for c in ordered if "bg" in provenance[c[0]] or "background" in provenance[c[0]]]
    if bg_candidates:
        background = bg_candidates[0][0]
    else:
        neutral_candidates = [c for c, _ in ordered if is_neutral(c)]
        background = neutral_candidates[0] if neutral_candidates else (secondary or primary or "#ffffff")
    if text is None:
        text_candidates = [c for c in ordered if re.search(r"text|fg|foreground", provenance[c[0]], re.I)]
        text = text_candidates[0][0] if text_candidates else choose_text_color(background or "#ffffff", "#000000")
    palette = {
        "primary": primary or background or "#000000",
        "secondary": secondary or primary or background or "#000000",
        "background": background or "#ffffff",
        "text": text,
        "accent": accent,
    }
    theme = "dark" if relative_luminance(parse_hex_color(palette["background"]) or (255, 255, 255)) < 0.35 else "light"
    return palette, theme


def compute_confidence(weighted_colors: List[Tuple[str, int, str]], palette: Dict[str, Optional[str]], has_theme_color: bool) -> float:
    score = 0.2
    primary = palette.get("primary")
    primary_score = 0
    secondary_score = 0
    source_types = set()
    for color, weight, source in weighted_colors:
        source_types.add(source)
        if color == primary:
            primary_score += weight
        if color == palette.get("secondary"):
            secondary_score += weight
    if primary and any(re.search(r"(primary|brand|main)", src, re.I) for _, _, src in weighted_colors if _ == primary):
        score += 0.3
    if has_theme_color:
        score += 0.2
    score += 0.1 * min(3, len(source_types))
    ratio = primary_score / (secondary_score + 1e-9)
    score += 0.2 * min(1.0, ratio / 3)
    return max(0.0, min(1.0, score))


def load_memory(memory_dir: str) -> Dict[str, object]:
    path = os.path.join(memory_dir, "extracted_palettes_history.json")
    if not os.path.exists(path):
        return {"latest_by_domain": {}, "history": []}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {"latest_by_domain": {}, "history": []}


def atomic_write(path: str, data: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
    os.replace(tmp, path)


def with_lock(memory_dir: str):
    lock_path = os.path.join(memory_dir, ".lock")
    class Lock:
        def __enter__(self):
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
            except FileExistsError:
                raise RuntimeError("Memory lock is held; try again later")
            return self
        def __exit__(self, exc_type, exc, tb):
            with contextlib.suppress(FileNotFoundError):
                os.remove(lock_path)
    return Lock()


def save_memory(memory_dir: str, domain: str, record: Dict[str, object]) -> None:
    os.makedirs(memory_dir, exist_ok=True)
    data = load_memory(memory_dir)
    data.setdefault("latest_by_domain", {})[domain] = record
    data.setdefault("history", []).append(record)
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    with with_lock(memory_dir):
        atomic_write(os.path.join(memory_dir, "extracted_palettes_history.json"), payload)


def extract_from_url(url: str, args: argparse.Namespace) -> Dict[str, object]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError("Only http/https URLs are allowed")
    html_bytes = fetch_url(url, timeout=args.timeout, max_bytes=args.max_html_bytes)
    html = html_bytes.decode("utf-8", errors="ignore")
    parser = collect_sources(html)
    weighted: List[Tuple[str, int, str]] = []
    sources_used = []
    for c in parser.meta_theme_color:
        hex_value = normalize_hex(c)
        if hex_value:
            weighted.append((hex_value, 6, "meta"))
            sources_used.append("meta")
    if parser.style_blocks:
        sources_used.append("style_block")
    for block in parser.style_blocks:
        for color, w in parse_colors_from_css(block):
            weighted.append((color, w, "style_block"))
    if parser.inline_styles:
        sources_used.append("inline")
        for color, w in parse_inline_styles(parser.inline_styles):
            weighted.append((color, w, "inline"))
    origin = parsed.hostname
    css_files = 0
    for href in parser.links:
        if css_files >= args.max_css_files:
            break
        resolved = urllib.parse.urljoin(url, href)
        parsed_css = urllib.parse.urlparse(resolved)
        if parsed_css.scheme not in {"http", "https"}:
            continue
        host = urllib.parse.urlparse(resolved).hostname
        if not host:
            continue
        if host != origin:
            if not args.allow_css_cdn:
                continue
            allowlist = {h.strip().lower() for h in args.css_host_allowlist.split(',')} if args.css_host_allowlist else set()
            if args.same_origin_css:
                continue
            if allowlist and host.lower() not in allowlist:
                continue
            if not allowlist:
                continue
        try:
            css_bytes = fetch_url(resolved, timeout=args.timeout, max_bytes=args.max_css_bytes)
            css_files += 1
            css_text = css_bytes.decode("utf-8", errors="ignore")
            sources_used.append("external_css")
            for color, w in parse_colors_from_css(css_text):
                weighted.append((color, w, "external_css"))
        except Exception as exc:  # noqa: PERF203
            logging.debug("Skipping CSS %s: %s", resolved, exc)
            continue
    palette, theme = select_palette(weighted)
    company = parser.meta_site_name or parser.meta_app_name or (parser.title or "").strip() or (origin or "")
    extracted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record = {
        "company": company,
        "url": url,
        "domain": origin,
        "palette": palette,
        "theme": theme,
        "confidence": compute_confidence(weighted, palette, bool(parser.meta_theme_color)),
        "sources_used": sorted(set(sources_used)),
        "extracted_at": extracted_at,
    }
    return record


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract brand palette from website")
    parser.add_argument("url")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--max-html-bytes", dest="max_html_bytes", type=int, default=2_000_000)
    parser.add_argument("--max-css-bytes", dest="max_css_bytes", type=int, default=1_000_000)
    parser.add_argument("--max-css-files", dest="max_css_files", type=int, default=5)
    parser.add_argument("--same-origin-css", dest="same_origin_css", action="store_true", default=True)
    parser.add_argument("--no-same-origin-css", dest="same_origin_css", action="store_false")
    parser.add_argument("--allow-css-cdn", dest="allow_css_cdn", action="store_true", default=False)
    parser.add_argument("--css-host-allowlist", dest="css_host_allowlist", default="")
    parser.add_argument("--memory-dir", dest="memory_dir", default=os.path.expanduser("~/Downloads/Pal/.pipeline_memory/slide_decks/"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    if args.quiet:
        level = logging.ERROR
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    domain = domain_from_url(args.url)
    memory = load_memory(args.memory_dir)
    cached = memory.get("latest_by_domain", {}).get(domain)
    if cached and not args.refresh:
        record = cached
    else:
        record = extract_from_url(args.url, args)
        try:
            save_memory(args.memory_dir, domain, record)
        except Exception as exc:  # noqa: PERF203
            logging.error("Failed to save memory: %s", exc)
    output_json = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            atomic_write(args.output, output_json)
        except Exception as exc:  # noqa: PERF203
            print(f"Error writing output: {exc}", file=sys.stderr)
            return 1
    else:
        print(output_json)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
