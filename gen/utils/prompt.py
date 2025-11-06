# gen/utils/prompt.py
from __future__ import annotations
import re
from typing import Tuple, List, Dict

# -------- utils --------
_WORD_RE = re.compile(r"[^\s,]+")


def _trim_to_words_csv(s: str, limit: int) -> str:
    words = re.findall(r"[^\s,]+(?:,)?", s)
    s = (
        " ".join(words[:limit]).strip(" ,")
        if len(words) > limit
        else " ".join(words).strip(" ,")
    )
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r",\s*,", ", ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s


def _dedupe_csv(s: str) -> str:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    seen, out = set(), []
    for p in parts:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return ", ".join(out)


# -------- classifiers --------
RX_TRANS = re.compile(
    r"\b(glass|clear|transparent|crystal|acrylic|plexi|chandelier|vase|cup)\b", re.I
)
RX_METAL = re.compile(
    r"\b(chrome|silver|polished|mirror|brass|bronze|stainless|titanium)\b", re.I
)
RX_CERAMIC = re.compile(r"\b(ceramic|porcelain|earthenware|stoneware)\b", re.I)
RX_LEATHER = re.compile(r"\b(leather)\b", re.I)
RX_FURN = re.compile(r"\b(chair|armchair|recliner|sofa|couch)\b", re.I)
RX_STONE = re.compile(r"\b(stone|gem|brooch|lapel|pin)\b", re.I)


# -------- backgrounds / materials / rims --------
def _bg(prompt: str) -> str:
    if RX_TRANS.search(prompt) or RX_STONE.search(prompt):
        # bright substrate helps gems/stone edges
        return "matte pure white seamless background (uniform, no texture)"
    if RX_METAL.search(prompt):
        return "matte pure white seamless background (uniform, no texture)"
    if RX_CERAMIC.search(prompt) or RX_LEATHER.search(prompt) or RX_FURN.search(prompt):
        return "matte mid-gray seamless background (uniform, no texture)"
    return "matte mid-gray seamless background (uniform, no texture)"


def _material(prompt: str) -> str:
    if RX_TRANS.search(prompt):
        return "clear material, controlled highlights, no caustics"
    if RX_METAL.search(prompt) or RX_STONE.search(prompt):
        return "polished mineral/metal, soft gradient reflections"
    if RX_CERAMIC.search(prompt):
        return "glazed ceramic, even glossy reflections"
    if RX_LEATHER.search(prompt):
        return "fine leather grain, soft sheen"
    if RX_FURN.search(prompt):
        return "upholstery seams sharp, wood frame clean"
    return "clean material, controlled highlights"


def _rim(prompt: str) -> str:
    if (
        RX_TRANS.search(prompt)
        or RX_CERAMIC.search(prompt)
        or RX_LEATHER.search(prompt)
        or RX_FURN.search(prompt)
    ):
        return "thin rim-light halo"
    if RX_STONE.search(prompt) or RX_METAL.search(prompt):
        return "subtle rim lights"
    return "subtle rim lights"


def _shadow(prompt: str) -> str:
    return "soft contact shadow" if not RX_TRANS.search(prompt) else "no shadow"


# -------- compact negatives --------
NEG_BASE = (
    "cluttered background, hard shadows, horizon line, gradients, texture, props, "
    "environment reflections, text, watermark, labels, people, hands, multiple objects, "
    "heavy glare, blown highlights, chromatic aberration, color cast, bokeh, motion blur, "
    "vignette, noise, dust, smudges, fingerprints"
)


def _neg(prompt: str) -> str:
    n = NEG_BASE
    if RX_TRANS.search(prompt):
        n += ", caustic patterns, background distortion from refraction"
    if RX_METAL.search(prompt) or RX_STONE.search(prompt):
        n += ", mirror-like environment reflections"
    if RX_FURN.search(prompt):
        # block interiors explicitly
        n += ", room interior, floor, ground plane, walls, corners, baseboards, rugs"
    return n


# -------- special overrides for the 4 failing prompts --------
SPECIAL: Dict[str, Dict] = {
    # regex pattern (lowercased) -> overrides
    r"\bstone brooch\b": {
        "bg": "matte pure white seamless background (uniform, no texture)",
        "material": "polished stone surface, crisp edges",
        "rim": "thin rim-light halo",
        "shadow": "soft contact shadow",
    },
    r"\brustic orange ceramic pot\b": {
        "bg": "matte charcoal seamless background (uniform, no texture)",  # stronger contrast than mid-gray
        "material": "glazed ceramic with even glossy reflections",
        "rim": "thin rim-light halo",
        "shadow": "soft contact shadow",
    },
    r"\bleather wallet\b": {
        "bg": "matte charcoal seamless background (uniform, no texture)",  # edge pop for brown/black leather
        "material": "fine leather grain, soft sheen",
        "rim": "bright thin rim-light halo",
        "shadow": "soft contact shadow",
    },
    r"\barmchair\b": {
        "bg": "matte mid-gray seamless background (uniform, no texture)",
        "material": "upholstery seams sharp, wood frame clean",
        "rim": "slightly stronger rim lights",
        "shadow": "soft contact shadow",
    },
}


def _apply_special(subject: str, vals: Dict[str, str]) -> Dict[str, str]:
    s = subject.lower()
    for pat, ov in SPECIAL.items():
        if re.search(pat, s):
            vals.update(ov)
    return vals


# -------- prompt builders --------
ESSENTIALS = "studio product shot, single object, centered"
LIGHTING = "soft light tent, diffuse softboxes, cross-polarized"


def _build_head(subject: str, bg: str, rim: str, shadow: str, material: str) -> str:
    parts = [
        subject,
        ESSENTIALS,
        LIGHTING,
        rim,
        f"on {bg}",
        shadow,
        "full object visible, clear silhouette",
        material,
        "isolated",  # tiny bias against scenes
    ]
    head = _dedupe_csv(", ".join(parts))
    # hard cap for CLIP (well under 77 tokens)
    head = _trim_to_words_csv(head, limit=40)
    return head


def tune_prompt(raw_prompt: str) -> Tuple[str, str]:
    """
    SD3.5-Turbo (CFG=0) prompt tuner.
    Returns (positive_prompt, negative_prompt).
    """
    subject = raw_prompt.strip()

    vals = {
        "bg": _bg(subject),
        "rim": _rim(subject),
        "shadow": _shadow(subject),
        "material": _material(subject),
    }
    vals = _apply_special(subject, vals)

    positive = _build_head(
        subject, vals["bg"], vals["rim"], vals["shadow"], vals["material"]
    )
    negative = _neg(subject)
    return positive, negative
