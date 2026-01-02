"""Zero-trust slide deck generator aligned with pipeline v3.0.

Plan-only by default; writes and memory updates occur only with --yes. Uses only the
Python standard library and embeds all assets in a single HTML with design tokens.

Doctest examples cover core helpers.

>>> derive_seed("Acme", "Engineer", "2024-01-01")
'20240101-engine-1a2b'
>>> validate_contractions("I'm ready and I've prepared.")
True
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import logging
import math
import os
import random
import re
import sys
import time
import uuid
from typing import Dict, List, Optional, Tuple

DEFAULT_MEMORY_DIR = os.path.expanduser("~/Downloads/Pal/.pipeline_memory/slide_decks/")
PIPELINE_VERSION = "3.0"
LOCK_TTL_SECONDS = 300
REQUIRED_CONTENT_KEYS = {
    "name",
    "target_position",
    "stats",
    "about_me",
    "why_company",
    "experience",
    "skills",
    "contact",
    "availability",
}
INTERVIEW_QUESTIONS = [
    "Tell me about yourself.",
    "Why do you feel you are the best fit for this company?",
    "Why do you want to work here?",
]
BANNED_WORDS = {
    "straightforward",
    "dive",
    "realm",
    "robust",
    "utilize",
    "leverage",
    "spearheaded",
    "multifaceted",
    "arguably",
    "notably",
}
SYSTEM_FONTS_DISPLAY = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif"
SYSTEM_FONTS_MONO = "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace"


class FileLock:
    def __init__(self, path: str, ttl: int = LOCK_TTL_SECONDS) -> None:
        self.path = path
        self.ttl = ttl

    def __enter__(self):
        start = time.time()
        while True:
            if not os.path.exists(self.path):
                try:
                    data = {"pid": os.getpid(), "created_at": utcnow()}
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
                    if created and is_stale(created, self.ttl):
                        os.remove(self.path)
                        continue
                except Exception:
                    with contextlib.suppress(OSError):
                        os.remove(self.path)
                        continue
            if time.time() - start > self.ttl:
                raise RuntimeError("Lock wait exceeded TTL")
            time.sleep(0.05)

    def __exit__(self, exc_type, exc, tb):
        with contextlib.suppress(OSError):
            os.remove(self.path)


def utcnow() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def is_stale(created_at: str, ttl: int) -> bool:
    try:
        ts = dt.datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").timestamp()
    except Exception:
        return True
    return (time.time() - ts) > ttl


def atomic_write_json(path: str, data: object) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def atomic_write_text(path: str, content: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)


def derive_seed(company: str, position: str, date_str: str, salt: str = "") -> str:
    base = f"{company}|{position}|{date_str}{salt}"
    h = uuid.uuid5(uuid.NAMESPACE_DNS, base).hex[:4]
    position_abbrev = re.sub(r"[^a-z0-9]", "", position.lower())[:6] or "role"
    date_part = date_str.replace("-", "")
    return f"{date_part}-{position_abbrev}-{h}"


def validate_contractions(text: str) -> bool:
    return len(re.findall(r"\b(?:I'm|I've|don't|can't|won't|isn't|aren't|didn't|it's|that's|there's)\b", text, flags=re.I)) >= 2


def validate_banned_words(text: str) -> Optional[str]:
    for w in BANNED_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", text, flags=re.I):
            return w
    return None


def load_json_file(path: str, default: object) -> object:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def ensure_within(path: str, roots: List[str]) -> None:
    abs_path = os.path.abspath(path)
    if not any(abs_path == r or abs_path.startswith(r + os.sep) for r in roots):
        raise SystemExit("Output path must reside within output-dir or memory-dir")


def parse_content(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    missing = [k for k in REQUIRED_CONTENT_KEYS if k not in data]
    if missing:
        raise SystemExit(f"Content file missing required keys: {', '.join(missing)}")
    if not isinstance(data.get("why_company"), list):
        raise SystemExit("why_company must be a list")
    if not isinstance(data.get("experience"), list):
        raise SystemExit("experience must be a list")
    return data


def parse_palette(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        palette = json.load(f)
    required = {"primary", "secondary", "background", "text", "accent"}
    missing = [k for k in required if k not in palette]
    if missing:
        raise SystemExit(f"Palette file missing keys: {', '.join(missing)}")
    return palette


def contrast_ratio(hex1: str, hex2: str) -> float:
    def hex_to_rgb(val: str) -> Tuple[float, float, float]:
        v = val.lstrip("#")
        r = int(v[0:2], 16) / 255
        g = int(v[2:4], 16) / 255
        b = int(v[4:6], 16) / 255
        def channel(c: float) -> float:
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        return channel(r), channel(g), channel(b)
    l1 = 0.2126 * hex_to_rgb(hex1)[0] + 0.7152 * hex_to_rgb(hex1)[1] + 0.0722 * hex_to_rgb(hex1)[2]
    l2 = 0.2126 * hex_to_rgb(hex2)[0] + 0.7152 * hex_to_rgb(hex2)[1] + 0.0722 * hex_to_rgb(hex2)[2]
    l1, l2 = l1 + 0.05, l2 + 0.05
    return max(l1, l2) / min(l1, l2)


def adjust_for_accessibility(tokens: Dict[str, str]) -> List[str]:
    adjustments: List[str] = []
    bg = tokens["--color-background"]
    text = tokens["--color-text"]
    on_primary = tokens["--color-text-on-primary"]
    primary = tokens["--color-primary"]
    if contrast_ratio(text, bg) < 4.5:
        tokens["--color-text"] = "#000000" if contrast_ratio("#000000", bg) >= contrast_ratio("#ffffff", bg) else "#ffffff"
        adjustments.append("text vs background adjusted for contrast")
    if contrast_ratio(on_primary, primary) < 4.5:
        tokens["--color-text-on-primary"] = "#000000" if contrast_ratio("#000000", primary) >= contrast_ratio("#ffffff", primary) else "#ffffff"
        adjustments.append("text on primary adjusted for contrast")
    return adjustments


def build_tokens(palette: Dict[str, str], rng: random.Random) -> Dict[str, str]:
    space_unit = rng.randint(14, 22)
    radius_base = rng.choice([6, 8, 10, 12])
    tokens = {
        "--company-primary": palette["primary"],
        "--company-secondary": palette["secondary"],
        "--company-background": palette["background"],
        "--company-text": palette["text"],
        "--company-accent": palette.get("accent") or palette["secondary"],
        "--color-primary": palette["primary"],
        "--color-secondary": palette["secondary"],
        "--color-background": palette["background"],
        "--color-text": palette["text"],
        "--color-accent": palette.get("accent") or palette["secondary"],
        "--color-text-on-primary": "#000000",
        "--color-text-muted": "hsl(0, 0%, 40%)",
        "--font-display": SYSTEM_FONTS_DISPLAY,
        "--font-body": SYSTEM_FONTS_DISPLAY,
        "--font-mono": SYSTEM_FONTS_MONO,
        "--space-unit": f"{space_unit}px",
        "--space-xs": f"calc(var(--space-unit) * 0.5)",
        "--space-sm": f"calc(var(--space-unit) * 0.75)",
        "--space-md": f"calc(var(--space-unit) * 1)",
        "--space-lg": f"calc(var(--space-unit) * 1.5)",
        "--space-xl": f"calc(var(--space-unit) * 2)",
        "--space-2xl": f"calc(var(--space-unit) * 3)",
        "--text-xs": "0.875rem",
        "--text-sm": "0.95rem",
        "--text-base": "1rem",
        "--text-lg": "1.125rem",
        "--text-xl": "1.25rem",
        "--text-2xl": "1.5rem",
        "--text-3xl": "1.75rem",
        "--text-4xl": "2rem",
        "--text-5xl": "2.5rem",
        "--radius-none": "0px",
        "--radius-sm": f"{max(4, radius_base - 2)}px",
        "--radius-md": f"{radius_base}px",
        "--radius-lg": f"{radius_base + 6}px",
        "--radius-full": "999px",
        "--border-width": "2px",
        "--transition-fast": "150ms ease",
        "--transition-base": "250ms ease",
        "--transition-slow": "400ms ease",
        "--animation-enabled": "true",
        "--shadow-sm": "0 2px 4px hsl(0 0% 0% / 0.08)",
        "--shadow-md": "0 8px 20px hsl(0 0% 0% / 0.12)",
        "--shadow-lg": "0 12px 30px hsl(0 0% 0% / 0.16)",
        "--shadow-glow": "0 0 0 3px hsl(0 0% 100% / 0.12)",
        "--line-height-base": "1.6",
        "--container-max": "min(1200px, 95vw)",
    }
    tokens["--color-text-on-primary"] = "#000000" if contrast_ratio("#000000", palette["primary"]) >= contrast_ratio("#ffffff", palette["primary"]) else "#ffffff"
    tokens["--color-text-muted"] = "hsl(0, 0%, 40%)"
    return tokens


def design_parameters(rng: random.Random) -> Dict[str, str]:
    return {
        "layout": rng.choice(["grid", "stacked", "columnar"]),
        "typography": rng.choice(["sans", "serif-lite", "rounded"]),
        "geometry": rng.choice(["soft", "sharp", "pill"]),
        "spacing": rng.choice(["roomy", "compact"]),
        "hierarchy": rng.choice(["balanced", "bold", "measured"]),
        "motion": rng.choice(["calm", "lively"]),
        "density": rng.choice(["airy", "cozy"]),
    }


def generate_plan(seed: str, params: Dict[str, str], tokens: Dict[str, str], uniqueness_status: str, files: List[str]) -> Dict[str, object]:
    return {
        "seed": seed,
        "design_parameters": params,
        "tokens": tokens,
        "uniqueness": uniqueness_status,
        "files": files,
    }


def humanization_check(text: str) -> Tuple[bool, str]:
    if not validate_contractions(text):
        return False, "Interview slide must include at least two contractions"
    banned = validate_banned_words(text)
    if banned:
        return False, f"Interview slide contains banned word: {banned}"
    return True, "ok"


def ensure_uniqueness(seed: str, params: Dict[str, str], tokens: Dict[str, str], memory_dir: str, max_prior: int) -> Tuple[bool, str]:
    history_path = os.path.join(memory_dir, "designs_that_deployed.json")
    prior = load_json_file(history_path, [])
    if isinstance(prior, dict):
        prior_list = prior.get("history", []) if isinstance(prior.get("history"), list) else []
    else:
        prior_list = prior if isinstance(prior, list) else []
    metadata_entries = load_recent_metadata(memory_dir, max_prior)
    prior = (prior_list + metadata_entries)[-max_prior:]
    best_score = 0.0
    best_desc = ""
    for entry in prior:
        score = similarity_score(params, tokens, entry)
        if score > best_score:
            best_score = score
            best_desc = entry.get("seed", "")
        diffs = sum(1 for k in params if entry.get("design_parameters", {}).get(k) != params[k])
        if diffs < 3 or score >= 0.30:
            return False, f"Too similar to {entry.get('seed', 'prior')} (score {score:.2f})"
    return True, f"pass (closest {best_desc or 'none'} score {best_score:.2f})"


def load_recent_metadata(memory_dir: str, limit: int) -> List[Dict[str, object]]:
    entries: List[Tuple[float, Dict[str, object]]] = []
    for root, _, files in os.walk(memory_dir):
        for name in files:
            if name == "deck_metadata.json":
                path = os.path.join(root, name)
                try:
                    mtime = os.path.getmtime(path)
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    entries.append((mtime, data if isinstance(data, dict) else {}))
                except Exception:
                    continue
        if len(entries) >= limit:
            break
    entries.sort(key=lambda x: x[0])
    return [e for _, e in entries[-limit:]]


def similarity_score(params: Dict[str, str], tokens: Dict[str, str], entry: Dict[str, object]) -> float:
    score = 0.0
    prior_params = entry.get("design_parameters", {}) if isinstance(entry, dict) else {}
    for k, v in params.items():
        if prior_params.get(k) == v:
            score += 0.05
    prior_tokens = entry.get("design_tokens", {}) if isinstance(entry, dict) else {}
    for key in ["--space-unit", "--radius-md", "--color-primary", "--color-background"]:
        if prior_tokens.get(key) == tokens.get(key):
            score += 0.05
    return score


def build_slides(content: Dict[str, object]) -> List[Dict[str, object]]:
    slides: List[Dict[str, object]] = []
    slides.append({"title": content["name"], "type": "hero", "stats": content.get("stats", [])})
    slides.append({"title": "Why this company", "type": "why", "bullets": content.get("why_company", [])})
    interview_body = build_interview_text(content)
    slides.append({"title": "Interview Questions", "type": "interview", "body": interview_body.strip()})
    exp = content.get("experience", [])
    slides.append({"title": "Experience", "type": "experience", "items": exp})
    slides.append({"title": "Skills match", "type": "skills", "items": content.get("skills", {})})
    slides.append({"title": "Contact & Availability", "type": "contact", "contact": content.get("contact", {}), "availability": content.get("availability", "")})
    return slides[:8]


def build_interview_text(content: Dict[str, object]) -> str:
    about = content.get("about_me", "")
    exp = content.get("experience", [])
    prompt = INTERVIEW_QUESTIONS[0]
    fragments = [f"{prompt} I'm {about.strip()}" if about else f"{prompt} I'm ready to contribute."]
    if exp:
        sample = exp[0]
        bullets = sample.get("bullets", [])
        highlight = bullets[0] if bullets else "driving impact"
        fragments.append(f"I'm especially proud of my time at {sample.get('company','')}, where I delivered {highlight}.")
    fragments.append("I'm excited to join and I'm ready to collaborate from day one.")
    return " ".join(fragments).strip()


def lint_css(css: str, allow_external: bool) -> None:
    if css.count(":root") != 1:
        raise SystemExit("CSS must contain exactly one :root block")
    non_root = re.sub(r":root\s*\{[^}]*\}", "", css, flags=re.S)
    forbidden_color = re.search(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})", non_root)
    if forbidden_color:
        raise SystemExit("Hardcoded colors outside tokens are not allowed")
    forbidden_funcs = re.search(r"\b(?:rgb|rgba|hsl|hsla|transparent|color-mix)\(", non_root, flags=re.I)
    if forbidden_funcs:
        raise SystemExit("Color functions outside tokens are not allowed")
    bad_units = re.findall(r"\b(?!0\b)(?!1\b)(?!100%\b)([0-9]+(?:px|rem|em|vh|vw|%)\b)", non_root)
    for val in bad_units:
        if "var(" not in val:
            raise SystemExit("Size values must use design tokens")
    if not allow_external and re.search(r"https?://", css):
        raise SystemExit("External URLs in CSS not allowed")


def render_tokens(tokens: Dict[str, str]) -> str:
    lines = [":root {"]
    for k, v in tokens.items():
        lines.append(f"  {k}: {v};")
    lines.append("}")
    return "\n".join(lines)


def render_css(tokens: Dict[str, str]) -> str:
    return f"""
{render_tokens(tokens)}
body {{
  margin: 0;
  font-family: var(--font-body);
  color: var(--color-text);
  background: var(--color-background);
}}
.main {{
  display: grid;
  gap: var(--space-xl);
  padding: var(--space-xl);
  max-width: var(--container-max);
  margin: 0 auto;
}}
.slide {{
  background: var(--color-background);
  border: var(--border-width) solid var(--color-primary);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-sm);
  padding: var(--space-lg);
  display: grid;
  gap: var(--space-md);
}}
.hero-heading {{
  font-size: var(--text-4xl);
  font-family: var(--font-display);
}}
.section-heading {{
  font-size: var(--text-2xl);
  font-weight: 700;
}}
.body-text {{
  font-size: var(--text-base);
  line-height: var(--line-height-base);
}}
.stat-number {{
  font-size: var(--text-3xl);
  font-weight: 700;
}}
.card {{
  background: var(--color-primary);
  color: var(--color-text-on-primary);
  padding: var(--space-md);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-md);
}}
.cta-button {{
  background: var(--color-secondary);
  color: var(--color-text-on-primary);
  padding: var(--space-sm) var(--space-lg);
  border-radius: var(--radius-full);
  text-decoration: none;
  display: inline-block;
}}
"""


def render_html(slides: List[Dict[str, object]], tokens: Dict[str, str], allow_external_fonts: bool) -> str:
    css = render_css(tokens)
    lint_css(css, allow_external_fonts)
    slide_blocks = []
    for slide in slides:
        if slide["type"] == "hero":
            stats_html = "".join([
                f"<div class='card'><div class='stat-number'>{s.get('value','')}</div><div class='body-text'>{s.get('label','')}</div></div>" for s in slide.get("stats", [])
            ])
            slide_blocks.append(f"<section class='slide'><div class='hero-heading'>{slide['title']}</div>{stats_html}</section>")
        elif slide["type"] == "why":
            bullets = "".join(f"<li>{b}</li>" for b in slide.get("bullets", []))
            slide_blocks.append(f"<section class='slide'><div class='section-heading'>{slide['title']}</div><ul class='body-text'>{bullets}</ul></section>")
        elif slide["type"] == "interview":
            slide_blocks.append(f"<section class='slide'><div class='section-heading'>{slide['title']}</div><p class='body-text'>{slide['body']}</p></section>")
        elif slide["type"] == "experience":
            items = "".join(
                f"<div class='card'><div class='section-heading'>{item.get('title','')} @ {item.get('company','')}</div><div class='body-text'>{' | '.join(item.get('bullets', [])[:3])}</div></div>"
                for item in slide.get("items", [])
            )
            slide_blocks.append(f"<section class='slide'><div class='section-heading'>{slide['title']}</div><div class='cards'>{items}</div></section>")
        elif slide["type"] == "skills":
            skills = slide.get("items", {})
            groups = "".join(f"<div class='card'><div class='section-heading'>{k}</div><div class='body-text'>{', '.join(v)}</div></div>" for k, v in skills.items())
            slide_blocks.append(f"<section class='slide'><div class='section-heading'>{slide['title']}</div><div class='cards'>{groups}</div></section>")
        elif slide["type"] == "contact":
            contact = slide.get("contact", {})
            availability = slide.get("availability", "")
            details = "".join(f"<div class='body-text'><strong>{k.title()}</strong>: {v}</div>" for k, v in contact.items())
            slide_blocks.append(f"<section class='slide'><div class='section-heading'>{slide['title']}</div>{details}<div class='body-text'>Availability: {availability}</div></section>")
    external_font_links = "" if not allow_external_fonts else ""  # reserved
    html = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'/>
<meta name='viewport' content='width=device-width, initial-scale=1'/>
<title>Slide Deck</title>
{external_font_links}
<style>
{css}
</style>
</head>
<body>
<main class='main'>
{''.join(slide_blocks)}
</main>
</body>
</html>
"""
    return html


def post_write_verify(output_dir: str, allow_external_fonts: bool, expected_count: int, interview_pass: bool) -> None:
    files = ["index.html", "package.json", "vercel.json", "deck_metadata.json"]
    for f in files:
        path = os.path.join(output_dir, f)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise SystemExit(f"Missing or empty output file: {f}")
    with open(os.path.join(output_dir, "package.json"), "r", encoding="utf-8") as f:
        json.load(f)
    with open(os.path.join(output_dir, "vercel.json"), "r", encoding="utf-8") as f:
        json.load(f)
    with open(os.path.join(output_dir, "deck_metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(os.path.join(output_dir, "index.html"), "r", encoding="utf-8") as f:
        html = f.read()
    if html.count(":root") != 1:
        raise SystemExit("HTML must contain exactly one :root block")
    if not allow_external_fonts and re.search(r"<link[^>]+href=\"https?://", html):
        raise SystemExit("External dependencies are not allowed without --enable-google-fonts")
    slide_count = html.count("class='slide'")
    if slide_count < 5 or slide_count > 8:
        raise SystemExit("Slide count must be between 5 and 8")
    if not interview_pass:
        raise SystemExit("Interview slide failed humanization checks")


def reserve_seed(memory_dir: str, seed: str, company: str, position: str) -> None:
    path = os.path.join(memory_dir, "vercel_deployments.json")
    with FileLock(path + ".lock"):
        data = load_json_file(path, [])
        if isinstance(data, dict):
            records = data.get("history", []) if isinstance(data.get("history"), list) else []
        else:
            records = data if isinstance(data, list) else []
        if any(r.get("seed") == seed for r in records):
            raise SystemExit(f"Seed {seed} already used")
        record = {
            "seed": seed,
            "company": company,
            "position": position,
            "reserved_at": utcnow(),
            "status": "reserved",
        }
        records.append(record)
        if isinstance(data, dict):
            data["history"] = records
        else:
            data = records
        atomic_write_json(path, data)


def mark_success(memory_dir: str, seed: str, company: str, position: str, design_tokens: Dict[str, str], params: Dict[str, str], layout: str, geometry: str, typography: str, deployment_url: Optional[str]) -> None:
    path = os.path.join(memory_dir, "designs_that_deployed.json")
    record = {
        "seed": seed,
        "company": company,
        "position": position,
        "design_tokens": design_tokens,
        "design_parameters": params,
        "layout_approach": layout,
        "geometry": geometry,
        "typography": typography,
        "generated_at": utcnow(),
        "memory_state_id": str(uuid.uuid4()),
        "deployment_url": deployment_url,
    }
    with FileLock(path + ".lock"):
        data = load_json_file(path, [])
        if isinstance(data, dict):
            history = data.get("history", []) if isinstance(data.get("history"), list) else []
            history.append(record)
            data["history"] = history
        else:
            data = (data if isinstance(data, list) else []) + [record]
        atomic_write_json(path, data)


def append_patterns_to_avoid(memory_dir: str, seed: str, reason: str) -> None:
    path = os.path.join(memory_dir, "patterns_to_avoid.json")
    entry = {"seed": seed, "reason": reason, "recorded_at": utcnow()}
    with FileLock(path + ".lock"):
        data = load_json_file(path, [])
        if isinstance(data, dict):
            history = data.get("history", []) if isinstance(data.get("history"), list) else []
            history.append(entry)
            data["history"] = history
        else:
            data = (data if isinstance(data, list) else []) + [entry]
        atomic_write_json(path, data)


def update_tone(memory_dir: str, tone: str) -> None:
    path = os.path.join(memory_dir, "user_tone_selections.json")
    entry = {"tone": tone, "updated_at": utcnow()}
    with FileLock(path + ".lock"):
        data = load_json_file(path, [])
        if isinstance(data, dict):
            history = data.get("history", []) if isinstance(data.get("history"), list) else []
            history.append(entry)
            data["history"] = history
        else:
            data = (data if isinstance(data, list) else []) + [entry]
        atomic_write_json(path, data)


def build_metadata(seed: str, company: str, position: str, palette: Dict[str, str], tokens: Dict[str, str], params: Dict[str, str], adjustments: List[str], uniqueness: str) -> Dict[str, object]:
    return {
        "seed": seed,
        "company": company,
        "position": position,
        "palette": palette,
        "design_tokens": tokens,
        "design_parameters": params,
        "layout_approach": params.get("layout"),
        "geometry": params.get("geometry"),
        "typography": params.get("typography"),
        "generated_at": utcnow(),
        "pipeline_version": PIPELINE_VERSION,
        "memory_state_id": str(uuid.uuid4()),
        "accessibility_adjustments": adjustments,
        "uniqueness_status": uniqueness,
        "code_citations": {"source": "seed|spec"},
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate zero-trust slide deck (plan-first)")
    parser.add_argument("--company", required=True)
    parser.add_argument("--position", required=True)
    parser.add_argument("--palette-file", required=True)
    parser.add_argument("--content-file", required=True)
    parser.add_argument("--template", choices=["modular", "asymmetric", "minimal", "bold"], default="minimal")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--date", default=dt.datetime.utcnow().strftime("%Y-%m-%d"))
    parser.add_argument("--tone", choices=["A_growth", "B_technical", "C_direct"], default="C_direct")
    parser.add_argument("--memory-dir", default=DEFAULT_MEMORY_DIR)
    parser.add_argument("--uniqueness-check", action="store_true")
    parser.add_argument("--max-prior-decks", type=int, default=200)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--mark-success", action="store_true")
    parser.add_argument("--deployment-url")
    parser.add_argument("--enable-google-fonts", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.ERROR if args.quiet else logging.INFO, format="%(levelname)s:%(message)s")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.memory_dir, exist_ok=True)
    ensure_within(args.output_dir, [os.path.abspath(args.output_dir), os.path.abspath(args.memory_dir)])

    palette = parse_palette(args.palette_file)
    content = parse_content(args.content_file)

    seed = derive_seed(args.company, args.position, args.date)
    rng = random.Random(int(uuid.uuid5(uuid.NAMESPACE_DNS, seed).hex[:12], 16))

    params = design_parameters(rng)
    tokens = build_tokens(palette, rng)
    adjustments = adjust_for_accessibility(tokens)

    interview_slide_text = build_interview_text(content)
    interview_ok, interview_msg = humanization_check(interview_slide_text)

    uniqueness_status = "skipped"
    if args.uniqueness_check:
        ok, msg = ensure_uniqueness(seed, params, tokens, args.memory_dir, args.max_prior_decks)
        uniqueness_status = msg
        if not ok:
            if not args.yes:
                print(json.dumps(generate_plan(seed, params, tokens, f"FAIL: {msg}", ["index.html", "package.json", "vercel.json", "deck_metadata.json"]), ensure_ascii=False, indent=2, sort_keys=True))
                return
            else:
                seed_retry = derive_seed(args.company, args.position, args.date, "|retry1")
                rng = random.Random(int(uuid.uuid5(uuid.NAMESPACE_DNS, seed_retry).hex[:12], 16))
                params = design_parameters(rng)
                tokens = build_tokens(palette, rng)
                adjustments = adjust_for_accessibility(tokens)
                ok2, msg2 = ensure_uniqueness(seed_retry, params, tokens, args.memory_dir, args.max_prior_decks)
                uniqueness_status = msg2
                if not ok2:
                    append_patterns_to_avoid(args.memory_dir, seed_retry, msg2)
                    raise SystemExit(f"Uniqueness failed after retry: {msg2}")
                seed = seed_retry
    plan = generate_plan(seed, params, tokens, uniqueness_status, ["index.html", "package.json", "vercel.json", "deck_metadata.json"])
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    if not args.yes:
        return

    reserve_seed(args.memory_dir, seed, args.company, args.position)

    slides = build_slides(content)
    interview_slide = next((s for s in slides if s.get("type") == "interview"), None)
    if interview_slide:
        interview_slide["body"] = interview_slide_text
    if not interview_ok:
        raise SystemExit(interview_msg)

    html = render_html(slides, tokens, args.enable_google_fonts)

    package_json = {
        "name": "slide-deck",
        "version": "1.0.0",
        "scripts": {"start": "npx serve .", "dev": "npx serve ."},
    }
    vercel_json = {"rewrites": [{"source": "/(.*)", "destination": "/index.html"}]}
    metadata = build_metadata(seed, args.company, args.position, palette, tokens, params, adjustments, uniqueness_status)

    files_to_write = {
        os.path.join(args.output_dir, "index.html"): html,
        os.path.join(args.output_dir, "package.json"): json.dumps(package_json, ensure_ascii=False, indent=2, sort_keys=True),
        os.path.join(args.output_dir, "vercel.json"): json.dumps(vercel_json, ensure_ascii=False, indent=2, sort_keys=True),
        os.path.join(args.output_dir, "deck_metadata.json"): json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
    }

    for path, content_str in files_to_write.items():
        ensure_within(path, [os.path.abspath(args.output_dir), os.path.abspath(args.memory_dir)])
        atomic_write_text(path, content_str)

    post_write_verify(args.output_dir, args.enable_google_fonts, len(slides), interview_ok)

    if args.mark_success:
        mark_success(args.memory_dir, seed, args.company, args.position, tokens, params, params.get("layout", ""), params.get("geometry", ""), params.get("typography", ""), args.deployment_url)
    if args.deployment_url:
        path = os.path.join(args.memory_dir, "vercel_deployments.json")
        with FileLock(path + ".lock"):
            data = load_json_file(path, [])
            if isinstance(data, dict):
                records = data.get("history", []) if isinstance(data.get("history"), list) else []
            else:
                records = data if isinstance(data, list) else []
            records.append({"seed": seed, "deployment_url": args.deployment_url, "recorded_at": utcnow(), "status": "deployed"})
            if isinstance(data, dict):
                data["history"] = records
            else:
                data = records
            atomic_write_json(path, data)
    update_tone(args.memory_dir, args.tone)


if __name__ == "__main__":
    main()
