"""Generate deployable slide decks deterministically.

This script plans and optionally builds a self-contained slide deck web app.

Doctest for seed derivation and token linting helpers:

>>> derive_seed("Acme", "Engineer", "2024-01-01")[:8]
'20240101'
>>> lint_css_for_tokens(':root{--color-primary:#fff;} body{color:var(--color-primary);}')
[]
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import random
import re
import sys
import textwrap
import time
import uuid
from typing import Dict, List, Optional, Tuple

PIPELINE_VERSION = "3.0"
DEFAULT_MEMORY_DIR = os.path.expanduser("~/Downloads/Pal/.pipeline_memory/slide_decks/")


def derive_seed(company: str, position: str, date_str: str, salt: str = "") -> str:
    """Derive deterministic seed key.

    >>> derive_seed("Acme", "Engineer", "2024-02-02")
    '20240202-engine-7b47'
    """
    position_abbrev = re.sub(r"[^a-z0-9]", "", position.lower())[:6] or "role"
    base = f"{company}|{position}|{date_str}{salt}"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:4]
    return f"{date_str.replace('-', '')}-{position_abbrev}-{digest}"


def build_rng(seed: str) -> random.Random:
    num = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12], 16)
    return random.Random(num)


def load_json(path: str, default: object) -> object:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return default


def atomic_write(path: str, content: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
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


def lint_css_for_tokens(css: str) -> List[str]:
    errors: List[str] = []
    root_blocks = re.findall(r":root\s*\{[^}]*\}", css)
    if len(root_blocks) != 1:
        errors.append("CSS must contain exactly one :root block")
    root_content = "".join(root_blocks)
    non_root = css.replace(root_content, "")
    forbidden = re.compile(r"#[0-9a-fA-F]{3,6}|rgb\(|hsl\(")
    for match in forbidden.finditer(non_root):
        snippet = non_root[max(0, match.start()-5): match.end()+5]
        if "var(" not in snippet:
            errors.append(f"Forbidden literal near '{snippet}'")
    if "var(" not in css:
        errors.append("CSS must use design tokens via var(--token)")
    return errors


def compute_contrast(fg: Tuple[int, int, int], bg: Tuple[int, int, int]) -> float:
    def lum(rgb: Tuple[int, int, int]) -> float:
        def channel(c: int) -> float:
            v = c / 255.0
            return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
        r, g, b = rgb
        return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
    l1, l2 = lum(fg), lum(bg)
    a, b = max(l1, l2), min(l1, l2)
    return (a + 0.05) / (b + 0.05)


def parse_palette(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    palette = data.get("palette") or data
    required = ["primary", "secondary", "background", "text"]
    for key in required:
        if key not in palette:
            raise ValueError(f"palette missing {key}")
    return palette


def select_tone(company: str, provided: Optional[str], memory_dir: str) -> str:
    if provided:
        return provided
    path = os.path.join(memory_dir, "user_tone_selections.json")
    data = load_json(path, {})
    return data.get(company) or "C_direct"


def gather_content(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs or [text]


def design_tokens(palette: Dict[str, str], rng: random.Random, template: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    base_space = rng.randint(14, 22)
    radius_base = rng.choice([4, 6, 8, 10])
    spacing = {
        "--space-unit": f"{base_space}px",
        "--space-xs": f"{base_space*0.5:.1f}px",
        "--space-sm": f"{base_space*0.75:.1f}px",
        "--space-md": f"{base_space}px",
        "--space-lg": f"{base_space*1.5:.1f}px",
        "--space-xl": f"{base_space*2:.1f}px",
        "--space-2xl": f"{base_space*2.5:.1f}px",
    }
    radius = {
        "--radius-sm": f"{radius_base}px",
        "--radius-md": f"{radius_base*2}px",
        "--radius-lg": f"{radius_base*3}px",
        "--border-width": "1.5px" if template in {"minimal", "asymmetric"} else "2px",
    }
    motion = {
        "--transition-fast": "120ms",
        "--transition-base": "200ms",
        "--transition-slow": "320ms",
        "--animation-enabled": "1" if rng.random() > 0.3 else "0",
    }
    typography_map = {
        "modular": ("Space Grotesk", "Inter", "Roboto Mono"),
        "asymmetric": ("Manrope", "Work Sans", "Roboto Mono"),
        "minimal": ("Nunito", "Source Sans Pro", "Fira Code"),
        "bold": ("Montserrat", "Lato", "IBM Plex Mono"),
    }
    display, body, mono = typography_map.get(template, typography_map["modular"])
    typography = {
        "--font-display": display,
        "--font-body": body,
        "--font-mono": mono,
        "--text-sm": "clamp(0.9rem, 1vw, 1rem)",
        "--text-md": "clamp(1rem, 1.2vw, 1.2rem)",
        "--text-lg": "clamp(1.2rem, 1.6vw, 1.6rem)",
        "--text-xl": "clamp(1.5rem, 2vw, 2rem)",
        "--text-3xl": "clamp(2.2rem, 3vw, 3rem)",
        "--text-5xl": "clamp(3rem, 4vw, 4rem)",
    }
    text_muted = "rgba(0,0,0,0.6)" if compute_contrast((0, 0, 0), hex_to_rgb(palette["background"])) > 4.5 else "rgba(255,255,255,0.7)"
    tokens = {
        "--company-primary": palette["primary"],
        "--company-secondary": palette["secondary"],
        "--company-background": palette["background"],
        "--company-text": palette["text"],
        "--company-accent": palette.get("accent") or palette["secondary"],
        "--color-primary": "var(--company-primary)",
        "--color-secondary": "var(--company-secondary)",
        "--color-background": "var(--company-background)",
        "--color-text": "var(--company-text)",
        "--color-accent": "var(--company-accent)",
        "--color-text-muted": text_muted,
        "--color-text-on-primary": optimal_text_on_primary(palette["primary"]),
    }
    tokens.update(spacing)
    tokens.update(radius)
    tokens.update(motion)
    tokens.update(typography)
    tokens.update({
        "--shadow-sm": "0 2px 4px rgba(0,0,0,0.08)",
        "--shadow-md": "0 6px 18px rgba(0,0,0,0.12)",
        "--shadow-lg": "0 12px 32px rgba(0,0,0,0.18)",
        "--shadow-glow": "0 0 0 3px var(--color-primary)",
    })
    adjustments: Dict[str, str] = {}
    text_contrast = compute_contrast(hex_to_rgb(tokens["--color-text"]) if tokens["--color-text"].startswith("#") else (0, 0, 0), hex_to_rgb(palette["background"]))
    if text_contrast < 4.5:
        better = "#000000" if compute_contrast((0, 0, 0), hex_to_rgb(palette["background"])) >= compute_contrast((255, 255, 255), hex_to_rgb(palette["background"])) else "#ffffff"
        tokens["--color-text"] = better
        adjustments["text"] = f"Adjusted to {better} for contrast"
    primary_rgb = hex_to_rgb(palette["primary"])
    on_primary = optimal_text_on_primary(palette["primary"])
    if compute_contrast(hex_to_rgb(on_primary), primary_rgb) < 4.5:
        fallback = "#000000" if compute_contrast((0, 0, 0), primary_rgb) >= compute_contrast((255, 255, 255), primary_rgb) else "#ffffff"
        tokens["--color-text-on-primary"] = fallback
        adjustments["text_on_primary"] = f"Adjusted to {fallback} for contrast"
    return tokens, adjustments


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def optimal_text_on_primary(primary: str) -> str:
    rgb = hex_to_rgb(primary)
    contrast_black = compute_contrast((0, 0, 0), rgb)
    contrast_white = compute_contrast((255, 255, 255), rgb)
    return "#000000" if contrast_black >= contrast_white else "#ffffff"


def generate_css(tokens: Dict[str, str], template: str) -> str:
    root_lines = [":root {"] + [f"  {k}: {v};" for k, v in sorted(tokens.items())] + ["}"]
    layout = {
        "modular": "display: grid; grid-template-columns: 1fr; gap: var(--space-lg);",
        "asymmetric": "display: grid; grid-template-columns: 3fr 2fr; gap: var(--space-lg);",
        "minimal": "display: flex; flex-direction: column; gap: var(--space-md);",
        "bold": "display: grid; grid-template-columns: 1fr; gap: var(--space-lg);",
    }[template]
    css = "\n".join(root_lines + [
        "body { background: var(--color-background); color: var(--color-text); font-family: var(--font-body); margin: 0; padding: var(--space-lg); }",
        "h1,h2,h3 { font-family: var(--font-display); margin: 0 0 var(--space-md); }",
        "p { margin: 0 0 var(--space-sm); line-height: 1.6; }",
        ".deck { max-width: 1080px; margin: 0 auto; }",
        ".slide { background: var(--color-background); border: var(--border-width) solid color-mix(in srgb, var(--color-primary) 30%, transparent); box-shadow: var(--shadow-md); padding: var(--space-lg); border-radius: var(--radius-lg); }",
        ".pill { display: inline-block; padding: calc(var(--space-xs)) calc(var(--space-sm)); border-radius: var(--radius-sm); background: var(--color-primary); color: var(--color-text-on-primary); }",
        ".grid { %s }" % layout,
        ".muted { color: var(--color-text-muted); }",
        "ul { padding-left: var(--space-lg); }",
    ])
    return css


def build_html(company: str, position: str, template: str, css: str, slides: List[Dict[str, str]]) -> str:
    google_fonts = {
        "modular": "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&family=Inter:wght@400;600&display=swap",
        "asymmetric": "https://fonts.googleapis.com/css2?family=Manrope:wght@400;700&family=Work+Sans:wght@400;600&display=swap",
        "minimal": "https://fonts.googleapis.com/css2?family=Nunito:wght@400;700&family=Source+Sans+Pro:wght@400;600&display=swap",
        "bold": "https://fonts.googleapis.com/css2?family=Montserrat:wght@500;800&family=Lato:wght@400;700&display=swap",
    }[template]
    slides_html = "".join(
        f"<section class=\"slide\"><h2>{s['title']}</h2><p>{s['body']}</p></section>" for s in slides
    )
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{company} – {position}</title>
  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
  <link href=\"{google_fonts}\" rel=\"stylesheet\">
  <style>{css}</style>
</head>
<body>
  <main class=\"deck\">
    <section class=\"slide\"><h1>{company}</h1><p class=\"muted\">Role: {position}</p></section>
    {slides_html}
  </main>
</body>
</html>
"""


def build_slides(company: str, position: str, content: List[str], rng: random.Random) -> List[Dict[str, str]]:
    slides: List[Dict[str, str]] = []
    slides.append({"title": "Why I fit", "body": f"Blending experience and curiosity to excel at {company} as {position}."})
    slides.append({"title": "Why {company}?", "body": f"Inspired by {company}'s mission; eager to contribute measurable impact."})
    about = content[0] if content else "Focused professional with a track record of delivery."
    slides.append({"title": "About", "body": about})
    for idx, para in enumerate(content[1:3], start=1):
        slides.append({"title": f"Experience {idx}", "body": para})
    slides.append({"title": "Availability", "body": "Open to interview discussions; ready to start soon."})
    return slides[:8]


def plan_summary(seed: str, template: str, tone: str, palette: Dict[str, str], output_dir: str, uniqueness: str) -> str:
    return textwrap.dedent(
        f"""
        Plan:
          seed: {seed}
          template: {template}
          tone: {tone}
          output: {output_dir}
          palette: primary={palette['primary']}, secondary={palette['secondary']}, background={palette['background']}
          uniqueness: {uniqueness}
        """
    ).strip()


def uniqueness_ok(current: Dict[str, str], prior: List[Dict[str, str]], max_sim: float = 0.3) -> bool:
    for entry in prior:
        matches = sum(1 for key in ["layout", "typography", "geometry", "spacing", "hierarchy", "motion", "density"] if entry.get(key) == current.get(key))
        similarity = matches / 7.0
        if matches > 4 or similarity >= max_sim:
            return False
    return True


def post_write_verify(output_dir: str) -> None:
    """Validate outputs exist, parse JSON, and enforce token usage."""

    files = ["index.html", "package.json", "vercel.json", "deck_metadata.json"]
    for name in files:
        path = os.path.join(output_dir, name)
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            raise RuntimeError(f"Missing expected file {name}")
    with open(os.path.join(output_dir, "package.json"), "r", encoding="utf-8") as f:
        json.load(f)
    with open(os.path.join(output_dir, "vercel.json"), "r", encoding="utf-8") as f:
        json.load(f)
    with open(os.path.join(output_dir, "deck_metadata.json"), "r", encoding="utf-8") as f:
        json.load(f)
    html = open(os.path.join(output_dir, "index.html"), "r", encoding="utf-8").read()
    if html.count(":root") != 1:
        raise RuntimeError(":root token block invalid")
    if "var(--" not in html:
        raise RuntimeError("Design tokens not used in HTML")


def write_outputs(output_dir: str, company: str, position: str, template: str, tokens: Dict[str, str], slides: List[Dict[str, str]], metadata: Dict[str, object]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    css = generate_css(tokens, template)
    css_errors = lint_css_for_tokens(css)
    if css_errors:
        raise RuntimeError("; ".join(css_errors))
    html = build_html(company, position, template, css, slides)
    pkg = {
        "name": f"deck-{company.lower().replace(' ', '-')}",
        "version": "1.0.0",
        "scripts": {"start": "npx serve .", "dev": "npx serve ."},
    }
    vercel = {"rewrites": [{"source": "/(.*)", "destination": "/index.html"}]}
    atomic_write(os.path.join(output_dir, "index.html"), html)
    atomic_write(os.path.join(output_dir, "package.json"), json.dumps(pkg, ensure_ascii=False, indent=2, sort_keys=True))
    atomic_write(os.path.join(output_dir, "vercel.json"), json.dumps(vercel, ensure_ascii=False, indent=2, sort_keys=True))
    atomic_write(os.path.join(output_dir, "deck_metadata.json"), json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    post_write_verify(output_dir)


def update_tone_memory(memory_dir: str, company: str, tone: str) -> None:
    os.makedirs(memory_dir, exist_ok=True)
    path = os.path.join(memory_dir, "user_tone_selections.json")
    data = load_json(path, {})
    data[company] = tone
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    with with_lock(memory_dir):
        atomic_write(path, payload)


def generate_metadata(seed: str, company: str, position: str, template: str, tone: str, palette: Dict[str, str], tokens: Dict[str, str], uniqueness_status: str) -> Dict[str, object]:
    return {
        "seed": seed,
        "company": company,
        "position": position,
        "template": template,
        "tone": tone,
        "palette": palette,
        "design_tokens": tokens,
        "layout_approach": template,
        "geometry": tokens["--radius-md"],
        "typography": tokens["--font-display"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pipeline_version": PIPELINE_VERSION,
        "memory_state_id": str(uuid.uuid4()),
        "accessibility_adjustments": {},
        "uniqueness_status": uniqueness_status,
    }


def uniqueness_signature(template: str, tokens: Dict[str, str]) -> Dict[str, str]:
    return {
        "layout": template,
        "typography": tokens.get("--font-display", ""),
        "geometry": tokens.get("--radius-md", ""),
        "spacing": tokens.get("--space-md", ""),
        "hierarchy": tokens.get("--text-3xl", ""),
        "motion": tokens.get("--animation-enabled", ""),
        "density": tokens.get("--space-unit", ""),
    }


def enforce_uniqueness(base_seed: str, company: str, position: str, date: str, template: str, palette: Dict[str, str], tokens: Dict[str, str], adjustments: Dict[str, str], memory_dir: str, max_prior: int) -> Tuple[str, random.Random, Dict[str, str], Dict[str, str], str]:
    """Run uniqueness gate; may retry once with salted seed."""

    prior_meta = load_json(os.path.join(memory_dir, "designs_that_deployed.json"), [])
    prior_meta = prior_meta[:max_prior]
    signature = uniqueness_signature(template, tokens)
    if uniqueness_ok(signature, prior_meta):
        return base_seed, build_rng(base_seed), tokens, adjustments, "passed"

    retry_seed = derive_seed(company, position, date, "|retry1")
    retry_rng = build_rng(retry_seed)
    retry_tokens, retry_adjustments = design_tokens(palette, retry_rng, template)
    if uniqueness_ok(uniqueness_signature(template, retry_tokens), prior_meta):
        return retry_seed, retry_rng, retry_tokens, retry_adjustments, "passed_after_retry"

    rejected_path = os.path.join(memory_dir, "patterns_to_avoid.json")
    data = load_json(rejected_path, [])
    data.append({"seed": retry_seed, "company": company, "position": position, "template": template})
    os.makedirs(memory_dir, exist_ok=True)
    with with_lock(memory_dir):
        atomic_write(rejected_path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit("Uniqueness check failed")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic slide decks")
    parser.add_argument("--company", required=True)
    parser.add_argument("--position", required=True)
    parser.add_argument("--palette-file", required=True)
    parser.add_argument("--content-file", required=True)
    parser.add_argument("--template", choices=["modular", "asymmetric", "minimal", "bold"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--date", default=time.strftime("%Y-%m-%d", time.gmtime()))
    parser.add_argument("--tone", choices=["A_growth", "B_technical", "C_direct"])
    parser.add_argument("--memory-dir", default=DEFAULT_MEMORY_DIR)
    parser.add_argument("--uniqueness-check", action="store_true")
    parser.add_argument("--max-prior-decks", type=int, default=200)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    if args.quiet:
        level = logging.ERROR
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    palette = parse_palette(args.palette_file)
    tone = select_tone(args.company, args.tone, args.memory_dir)
    seed = derive_seed(args.company, args.position, args.date)
    rng = build_rng(seed)
    content = gather_content(args.content_file)
    tokens, adjustments = design_tokens(palette, rng, args.template)
    slides = build_slides(args.company, args.position, content, rng)
    uniqueness_status = "skipped"
    if args.yes and args.uniqueness_check:
        seed, rng, tokens, adjustments, uniqueness_status = enforce_uniqueness(
            seed,
            args.company,
            args.position,
            args.date,
            args.template,
            palette,
            tokens,
            adjustments,
            args.memory_dir,
            args.max_prior_decks,
        )
        slides = build_slides(args.company, args.position, content, rng)

    plan = plan_summary(seed, args.template, tone, palette, args.output_dir, uniqueness_status)
    print(plan)

    if not args.yes:
        return 0

    metadata = generate_metadata(seed, args.company, args.position, args.template, tone, palette, tokens, uniqueness_status)
    metadata["accessibility_adjustments"] = adjustments
    write_outputs(args.output_dir, args.company, args.position, args.template, tokens, slides, metadata)
    update_tone_memory(args.memory_dir, args.company, tone)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
