from __future__ import annotations
import re
from typing import Dict, List, Tuple

# ---------- Core negative prompt tuned for clean edges / easy matting ----------
NEGATIVE_BASE = (
    "cluttered background, hard shadows, horizon line, gradients, texture, "
    "busy environment, props, reflections of studio gear, environment reflections, "
    "text, watermark, logo, label, people, hands, multiple objects, "
    "heavy glare, blown highlights, chromatic aberration, color cast, "
    "bokeh, motion blur, vignette, noise, dust, smudges, fingerprints"
)

# ---------- Keyword groups ----------
KW = {
    "transparent": [
        r"\bglass\b",
        r"\bcrystal\b",
        r"\bclear\b",
        r"\btranslucent\b",
        r"\btransparent\b",
        r"\bice\b",
        r"\bacrylic\b",
        r"\bplexi\b",
        r"\bchandelier\b",
        r"\bvase\b",
        r"\bclear cup\b",
        r"\bglass(es)?\b",
    ],
    "metal_polished": [
        r"\bchrome\b",
        r"\bsilver\b",
        r"\bpolished\b",
        r"\bmirror(ed)?\b",
        r"\bbrass\b",
        r"\bbronze\b",
        r"\bstainless\b",
        r"\bmetal\b",
        r"\btitanium\b",
        r"\btrombone\b",
        r"\btrumpet\b",
    ],
    "gemstone": [
        r"\bdiamond\b",
        r"\btopaz\b",
        r"\bruby\b",
        r"\bagate\b",
        r"\bpearl\b",
        r"\bjade\b",
        r"\bquartz\b",
    ],
    "ceramic_porcelain": [
        r"\bporcelain\b",
        r"\bceramic\b",
        r"\bivory\b",
        r"\balabaster\b",
        r"\bstatue\b",
        r"\bfigurine\b",
    ],
    "fabric_soft": [
        r"\bscarf\b",
        r"\bcloth\b",
        r"\bgloves?\b",
        r"\bvelvet\b",
        r"\bwool(en)?\b",
        r"\btie-?dye\b",
    ],
    "plastic": [r"\bplastic\b", r"\bstorage bin\b", r"\bvisor\b"],
    "wood": [
        r"\bwood(en)?\b",
        r"\bmahogany\b",
        r"\blute\b",
        r"\btable\b",
        r"\bchair\b",
        r"\bnesting\b",
        r"\bbookcase\b",
    ],
    "leather": [r"\bleather\b", r"\bwallet\b", r"\brecliner\b", r"\bwingback\b"],
    "food": [r"\byogurt\b", r"\bdonut\b", r"\bchicken\b", r"\blemonade\b"],
    "plant": [
        r"\bflower\b",
        r"\bplant\b",
        r"\brosemary\b",
        r"\banemone\b",
        r"\bcoral\b",
        r"\bfungus\b",
        r"\bpansy\b",
        r"\bwisteria\b",
        r"\belderflower\b",
    ],
    "animal_creature": [
        r"\bbear\b",
        r"\bbull\b",
        r"\bkitten\b",
        r"\bpoodle\b",
        r"\bseahorse\b",
        r"\bpeacock\b",
        r"\belephant\b",
    ],
    "character_fig": [
        r"\bmermaid\b",
        r"\bfairy\b",
        r"\bunicorn\b",
        r"\bgoblin\b",
        r"\bminotaur\b",
        r"\bgremlin\b",
        r"\brobot\b",
        r"\balien\b",
        r"\bprincess\b",
    ],
    "vehicle_tool": [
        r"\bmotorcycle\b",
        r"\bforklift\b",
        r"\bjeep\b",
        r"\bspade\b",
        r"\bwrench(es)?\b",
        r"\bcompressor\b",
        r"\btape measure\b",
        r"\bscrewdriver\b",
        r"\bbearing\b",
    ],
    "instrument": [r"\bguitar\b", r"\bukulele?\b", r"\baccordion\b"],
    "jewelry": [
        r"\bnecklace\b",
        r"\bpendant\b",
        r"\bbrooch\b",
        r"\bring\b",
        r"\bcrown\b",
        r"\bscepter\b",
        r"\bjewelry\b",
    ],
    "architectural": [r"\bstair(case)?\b", r"\bramp\b", r"\bpavilion\b", r"\bcastle\b"],
    "toy": [r"\btoy\b", r"\baction figure\b", r"\bbouncing ball\b", r"\bomozo\b"],
    "scientific_mech": [
        r"\bturbine\b",
        r"\bcompressor wheel\b",
        r"\binchw(heel)?\b",
        r"\bgear\b",
    ],
    "container": [r"\bcup\b", r"\bvase\b", r"\bbin\b", r"\bbox\b", r"\binkwell\b"],
}

# Rank of backgrounds by subject type:
# - transparent: charcoal/mid-gray to maximize rim contrast
# - polished metals/gems: pure white to control reflections
# - most others: mid-gray
BG_PICK = {
    "transparent": "seamless matte charcoal background (uniform, no texture)",
    "metal_polished": "seamless matte pure white background (uniform, no texture)",
    "gemstone": "seamless matte pure white background (uniform, no texture)",
    "default": "seamless matte mid-gray background (uniform, no texture)",
}

# Lighting presets
LIGHT_TENT = (
    "in a soft light tent with 360° diffuse softboxes, cross-polarized to reduce glare"
)
RIM_LIGHT_SUBTLE = "subtle dual rim lights outlining the silhouette"
RIM_LIGHT_STRONG = (
    "thin bright rim light halo along the object perimeter for edge separation"
)

# Camera/composition preset
CAMERA_COMPOSE = "studio product photo, single object, centered composition, elevation 15°, 70mm, f/16, ultra-sharp focus"

# Shadow control for BiRefNet (soft or none)
SHADOW_SOFT = "soft contact shadow only"
SHADOW_NONE = "shadow-free"

# Material descriptors to encourage correct rendering
MATERIAL_HINTS = {
    "transparent": "clear, high-clarity material with crisp edges; no caustics; controlled highlights",
    "metal_polished": "polished metal with controlled reflections; soft gradient highlights; micro-scratches (subtle)",
    "gemstone": "faceted gemstone with crisp facets and controlled spectral sparkle (not blown out)",
    "ceramic_porcelain": "glossy ceramic/porcelain with soft, even reflections",
    "fabric_soft": "accurate textile fibers, minimal fuzz and no stray threads",
    "plastic": "matte to satin plastic surface without warping",
    "wood": "clean wood grain, no props",
    "leather": "fine leather grain, soft sheen, no creases beyond natural shape",
    "food": "clean, appetizing surface without mess; no plate unless specified",
    "plant": "crisp leaf edges, no soil spill, neutral pot if any",
    "animal_creature": "single figurative subject, full silhouette visible",
    "character_fig": "single figurative subject, clean silhouette, no scene background",
    "vehicle_tool": "product-style view, no environment, clean edges",
    "instrument": "clean reflections, full silhouette, strings/details sharp",
    "jewelry": "macro-friendly detail, clean prongs/settings, controlled sparkle",
    "architectural": "isolated object view, not a scene; full silhouette",
    "toy": "clean molded details, no packaging",
    "scientific_mech": "machined surfaces, crisp edges, no motion blur",
    "container": "clear wall thickness and lip detail if transparent",
}

# Compile regex patterns once
PATTERNS: List[Tuple[str, re.Pattern]] = []
for label, patterns in KW.items():
    PATTERNS.append((label, re.compile("|".join(patterns), flags=re.IGNORECASE)))


def _classify(prompt: str) -> List[str]:
    hits = []
    for label, rx in PATTERNS:
        if rx.search(prompt):
            hits.append(label)
    return hits


def _pick_background(labels: List[str]) -> str:
    if "transparent" in labels:
        return BG_PICK["transparent"]
    if "metal_polished" in labels or "gemstone" in labels:
        return BG_PICK["metal_polished"]
    return BG_PICK["default"]


def _material_sentence(labels: List[str]) -> str:
    # Prioritize strongest cues
    priority = [
        "transparent",
        "metal_polished",
        "gemstone",
        "ceramic_porcelain",
        "fabric_soft",
        "plastic",
        "wood",
        "leather",
        "food",
        "plant",
        "animal_creature",
        "character_fig",
        "vehicle_tool",
        "instrument",
        "jewelry",
        "architectural",
        "toy",
        "scientific_mech",
        "container",
    ]
    for p in priority:
        if p in labels:
            return MATERIAL_HINTS[p]
    return "clean surface rendering with controlled highlights"


def _shadow_mode(labels: List[str]) -> str:
    # For transparent/glass, shadow-free often helps matting; others use soft shadow
    return SHADOW_NONE if "transparent" in labels else SHADOW_SOFT


def _extra_edges(labels: List[str]) -> str:
    # Stronger rim for transparent or very dark subjects (heuristic)
    if "transparent" in labels:
        return RIM_LIGHT_STRONG
    # Dark objects that often melt into background
    darkish = [r"\bblack\b", r"\bmidnight\b", r"\bcharcoal\b", r"\bdeep (blue|brown)\b"]
    if any(
        re.search(rx, "", re.I) for rx in darkish
    ):  # placeholder, could inspect the prompt if needed
        return RIM_LIGHT_STRONG
    return RIM_LIGHT_SUBTLE


def tune_prompt(raw_prompt: str) -> Tuple[str, str]:
    """
    Returns a dict with keys:
      - prompt: tuned positive prompt
      - negative: negative prompt
    Strategy:
      * force studio product look (light tent, cross-polarized, rim lights)
      * pick background based on material (charcoal for transparent; white for polished metal/gem; mid-gray otherwise)
      * set camera & composition for clear silhouette
      * control shadows (soft or none) to help BiRefNet
      * keep original concept up front
    """
    p = raw_prompt.strip()

    # Heuristic: ensure we phrase it as a single, centered product shot
    if not re.search(r"\b(product|studio) photo\b", p, re.I):
        concept = p
        lead = f"{concept}, studio product photo"
    else:
        lead = p

    labels = _classify(p)
    bg = _pick_background(labels)
    material_line = _material_sentence(labels)
    shadow = _shadow_mode(labels)
    rim = _extra_edges(labels)

    positive = (
        f"{lead}, {CAMERA_COMPOSE}, {LIGHT_TENT}, {rim}, "
        f"on {bg}, {shadow}, {material_line}, "
        "no props, full object in frame, clean silhouette, "
        "controlled highlights, minimal specular hotspots"
    )

    # Add a bit of img2img friendliness if you run refinement
    # (safe to include in txt2img; models usually ignore unknown tokens)
    positive += ", match base composition, preserve proportions"

    # Negative prompt
    negative = NEGATIVE_BASE

    # Small special cases to de-confuse the model for specific materials
    if "transparent" in labels:
        positive += ", no caustics, no internal clutter"
        negative += ", refractions causing background distortion, caustic patterns"
    if "metal_polished" in labels or "gemstone" in labels:
        positive += ", soft gradient reflections, anti-glare diffusion"
        negative += ", mirror-like environment reflections"

    return positive, negative
