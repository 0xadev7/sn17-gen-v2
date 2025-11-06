from __future__ import annotations
import re
from typing import Tuple


# --- helpers ---------------------------------------------------------------
def _wcount(s: str) -> int:
    return 0 if not s else len(re.findall(r"[^\s,]+", s))


def _trim_to_words(s: str, limit: int) -> str:
    if not s:
        return s
    words = re.findall(r"[^\s,]+(?:,)?", s)  # keep commas stuck to words
    if len(words) <= limit:
        return " ".join(words).strip()
    return " ".join(words[:limit]).strip().rstrip(",")


def _clean_commas(s: str) -> str:
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r",\s*,", ", ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip(" ,")


# --- material/background heuristics ----------------------------------------
_TRANS_RX = re.compile(
    r"\b(glass|crystal|clear|transparent|acrylic|plexi|chandelier|vase)\b", re.I
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
        return "clear high-clarity material, controlled highlights, no caustics"
    if _METAL_RX.search(prompt):
        return "polished metal with soft gradient reflections, anti-glare diffusion"
    return "clean material rendering with controlled highlights"


def _shadow_for(prompt: str) -> str:
    return "no shadow" if _TRANS_RX.search(prompt) else "soft contact shadow only"


def _rim_for(prompt: str) -> str:
    return (
        "thin rim-light halo along the silhouette"
        if _TRANS_RX.search(prompt)
        else "subtle rim lights outlining the silhouette"
    )


# --- main -------------------------------------------------------------------
NEGATIVE_COMPACT = (
    "cluttered background, hard shadows, horizon line, gradients, texture, props, "
    "environment reflections, text, watermark, labels, people, hands, multiple objects, "
    "heavy glare, blown highlights, chromatic aberration, color cast, bokeh, motion blur, "
    "vignette, noise, dust, smudges, fingerprints"
)


def tune_prompt(raw_prompt: str) -> Tuple[str, str]:
    """
    Returns (positive_prompt, negative_prompt)
    Strategy:
      1) Put CLIP-critical clauses FIRST (<= ~70 words).
      2) Move nice-to-have detail AFTER that (T5 can still use it).
      3) Compact negative prompt.
    """
    subject = raw_prompt.strip()

    bg = _bg_for(subject)
    rim = _rim_for(subject)
    shadow = _shadow_for(subject)
    material = _material_line(subject)

    # --- HEAD (CLIP must see these) ~ prioritized, concise ---
    head_parts = [
        subject,  # core concept first
        "studio product shot, single object, centered",
        "elevation 15°, 70mm, f/16, ultra-sharp focus",
        "soft light tent, diffuse softboxes, cross-polarized",
        rim,
        f"on {bg}",
        shadow,
        "full object in frame, clear silhouette",
        material,
    ]
    head = _clean_commas(", ".join(head_parts))

    # Ensure head stays within ~70 words for CLIP encoders
    head = _trim_to_words(head, limit=70)

    # --- TAIL (T5 sees this; CLIP may truncate before it) ---
    tail_parts = [
        "controlled highlights, low specular hotspots",
        "no props",
        "match composition, keep proportions",
    ]
    # Transparent tweak
    if _TRANS_RX.search(subject):
        tail_parts.append("no internal clutter")
    # Metal/gem tweak already encoded in material

    tail = _clean_commas(", ".join(tail_parts))

    positive = _clean_commas(f"{head}, {tail}")

    negative = NEGATIVE_COMPACT
    if _TRANS_RX.search(subject):
        negative += ", caustic patterns, background distortion from refraction"
    if _METAL_RX.search(subject):
        negative += ", mirror-like environment reflections"

    return positive, negative
