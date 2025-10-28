from __future__ import annotations
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageStat, ImageFilter


def _alpha_coverage_ratio(rgba: Image.Image) -> float:
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    alpha = np.asarray(rgba.split()[-1], dtype=np.float32) / 255.0
    return float(alpha.mean())


def _sharpness_score(img: Image.Image) -> float:
    # quick Laplacian-ish focus metric
    g = img.convert("L").filter(ImageFilter.FIND_EDGES)
    stat = ImageStat.Stat(g)
    return float(stat.var[0] ** 0.5)


def score_views(rgba_views: List[Image.Image]) -> List[Tuple[int, float]]:
    """
    Score each view by (alpha coverage * sharpness).
    Returns list of (index, score) sorted desc.
    """
    scored = []
    for i, im in enumerate(rgba_views):
        try:
            cov = _alpha_coverage_ratio(im)
            shp = _sharpness_score(im)
            scored.append((i, float(cov * (0.5 + shp))))
        except Exception:
            scored.append((i, 0.0))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored
