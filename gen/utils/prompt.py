from __future__ import annotations
import re
from typing import Tuple


# ---------- tiny utils ----------


def _trim_to_words(s: str, limit: int) -> str:
    if not s:
        return s
    words = re.findall(r"[^\s,]+(?:,)?", s)
    if len(words) <= limit:
        return " ".join(words).strip().strip(", ")
    return " ".join(words[:limit]).strip().strip(", ")


def _dedupe_phrases(s: str) -> str:
    # remove exact dup phrases split by commas
    parts = [p.strip() for p in s.split(",") if p.strip()]
    seen, out = set(), []
    for p in parts:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return ", ".join(out)


def _clean_commas(s: str) -> str:
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r",\s*,", ", ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip(" ,")


# ---------- quick material/background heuristics ----------
_TRANS_RX = re.compile(
    r"\b(glass|crystal|clear|transparent|acrylic|plexi|chandelier|vase|cup)\b", re.I
)
_METAL_RX = re.compile(
    r"\b(chrome|silver|polished|mirror|brass|bronze|stainless|titanium|trombone|trumpet)\b",
    re.I,
)


def _bg_for(prompt: str) -> str:
    if _TRANS_RX.search(prompt):
        return "matte charcoal seamless background (uniform, no texture)"
    if _METAL_RX.search(prompt):
        return "matte pure white seamless background (uniform, no texture)"
    return "matte mid-gray seamless background (uniform, no texture)"


def _material_line(prompt: str) -> str:
    if _TRANS_RX.search(prompt):
        return "clear material, controlled highlights, no caustics"
    if _METAL_RX.search(prompt):
        return "polished metal, soft gradient reflections"
    return "clean material, controlled highlights"


def _shadow_for(prompt: str) -> str:
    return "no shadow" if _TRANS_RX.search(prompt) else "soft contact shadow"


def _rim_for(prompt: str) -> str:
    return "thin rim-light halo" if _TRANS_RX.search(prompt) else "subtle rim lights"


# ---------- compact negative ----------
NEGATIVE_COMPACT = (
    "cluttered background, hard shadows, horizon line, gradients, texture, props, "
    "environment reflections, text, watermark, labels, people, hands, multiple objects, "
    "heavy glare, blown highlights, chromatic aberration, color cast, bokeh, motion blur, "
    "vignette, noise, dust, smudges, fingerprints"
)


# ---------- main API ----------
def tune_prompt(raw_prompt: str) -> Tuple[str, str]:
    """
    Returns (positive_prompt, negative_prompt).
    Puts CLIP-critical constraints in the first ~55 words to avoid 77-token truncation.
    """
    subject = raw_prompt.strip()

    bg = _bg_for(subject)
    rim = _rim_for(subject)
    shadow = _shadow_for(subject)
    material = _material_line(subject)

    # HEAD (CLIP-visible). Keep ultra concise; no fluff; no repeats.
    head_parts = [
        subject,  # core concept
        "studio product shot, single object, centered",
        "elevation 15°, 70mm, f/16, ultra-sharp focus",
        "soft light tent, diffuse softboxes, cross-polarized",
        rim,
        f"on {bg}",
        shadow,
        "full object in frame, clear silhouette",
        material,
    ]
    head = _dedupe_phrases(_clean_commas(", ".join(head_parts)))
    head = _trim_to_words(head, limit=55)  # strict cap for CLIP

    # TAIL (T5 benefits; CLIP may not see this)
    tail_parts = [
        "low specular hotspots",
        "no props",
        "match composition, keep proportions",
    ]
    if _TRANS_RX.search(subject):
        tail_parts.append("no internal clutter")
    tail = _dedupe_phrases(_clean_commas(", ".join(tail_parts)))

    positive = _clean_commas(f"{head}, {tail}")

    # Negative (compact + material addenda)
    negative = NEGATIVE_COMPACT
    if _TRANS_RX.search(subject):
        negative += ", caustic patterns, background distortion from refraction"
    if _METAL_RX.search(subject):
        negative += ", mirror-like environment reflections"

    return positive, negative
