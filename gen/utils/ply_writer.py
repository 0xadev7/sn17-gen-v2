import io
import math
from typing import Optional, Union, Tuple

import numpy as np
import meshio
import trimesh

# Constants matched to your PlyLoader
_SH_C0 = 0.28209479177387814
_MEAN_DC = 0.5  # features_dc = 0.5 when f_dc_* == 0.0 in your loader


def _logit(p: float) -> float:
    """Return log(p/(1-p)) with guards."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1.0 - p))


def _rgb_to_fdc(rgb01: np.ndarray) -> np.ndarray:
    """
    Convert desired DC color in [0,1] to f_dc_* so that
    features_dc = MEAN + SH_C0 * f_dc => equals desired color.
    f_dc = (rgb01 - MEAN)/SH_C0
    """
    return (rgb01 - _MEAN_DC) / _SH_C0


def _extract_vertex_rgb01(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
    """
    Try to extract per-vertex colors as float [0,1].
    Returns None if unavailable.
    """
    # 1) direct vertex_colors
    if hasattr(mesh, "visual") and mesh.visual is not None:
        # Per-vertex colors
        if (
            hasattr(mesh.visual, "vertex_colors")
            and mesh.visual.vertex_colors is not None
        ):
            c = np.asarray(mesh.visual.vertex_colors)
            if c.size > 0:
                c = c[:, :3].astype(np.float32) / 255.0
                return c
        # A single face/mesh color broadcast to vertices
        if hasattr(mesh.visual, "to_color"):
            vc = mesh.visual.to_color().vertex_colors
            if vc is not None and len(vc) == len(mesh.vertices):
                c = vc[:, :3].astype(np.float32) / 255.0
                return c
    return None


def write_gaussians_ply_from_trimesh(
    mesh: trimesh.Trimesh,
    target: Union[str, io.BytesIO],
    *,
    # Sampling: if n_samples is provided, sample points on the surface; else use vertices
    n_samples: Optional[int] = None,
    sample_method: str = "even",  # "even" or "random"
    # Gaussian attribute defaults
    default_opacity: float = 0.9,  # in [0,1]; stored as logit, PlyLoader applies sigmoid
    default_scale: Union[float, Tuple[float, float, float]] = 0.01,  # meters-ish
    default_rotation_quat: Tuple[float, float, float, float] = (
        1.0,
        0.0,
        0.0,
        0.0,
    ),  # w,x,y,z
    # Color: if mesh has per-vertex color, it will be used; else this fallback
    default_rgb01: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> None:
    """
    Create a PLY (meshio) with point_data fields required by PlyLoader:
      - 'opacity'            (float32)   -> stored as logits; loader applies sigmoid
      - 'rot_0'..'rot_3'     (float32)   -> quaternion; loader normalizes
      - 'scale_0'..'scale_2' (float32)   -> exp() is applied in loader
      - 'f_dc_0'..'f_dc_2'   (float32)   -> used to compute features_dc
    Positions use sampled points or mesh vertices.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    target : filepath (str) or io.BytesIO
    n_samples : number of surface points (if None, uses vertices)
    sample_method : 'even' or 'random' (passed to trimesh sampling utilities)
    default_opacity : in [0,1], will be converted to logit before writing
    default_scale : isotropic float or (sx,sy,sz)
    default_rotation_quat : (w, x, y, z)
    default_rgb01 : used if no per-vertex color is available
    """
    # 1) Get positions
    if n_samples is not None and n_samples > 0 and mesh.area > 0:
        if sample_method == "even":
            pts, _ = trimesh.sample.sample_surface_even(mesh, n_samples)
        else:
            pts, _ = trimesh.sample.sample_surface(mesh, n_samples)
    else:
        pts = np.asarray(mesh.vertices, dtype=np.float64)
        if pts.size == 0:
            raise ValueError("Mesh has no vertices and no sampling was requested.")

    n = pts.shape[0]

    # 2) Prepare attributes
    # Opacity stored as logits (PlyLoader: sigmoid on read)
    opacity_logit = np.full((n,), _logit(default_opacity), dtype=np.float32)

    # Quaternion (w,x,y,z), normalized later by PlyLoader anyway
    rot = np.tile(np.asarray(default_rotation_quat, dtype=np.float32), (n, 1))
    rot_0, rot_1, rot_2, rot_3 = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]

    # Scales: PlyLoader will do exp() -> so we should store natural-log scales.
    # If the caller provided linear scale(s), convert to log for storage.
    def _as_log_scales(s):
        s = np.asarray(s, dtype=np.float32)
        if s.ndim == 0:  # isotropic
            s = np.array([s, s, s], dtype=np.float32)
        if np.any(s <= 0):
            raise ValueError("Scales must be positive to take log.")
        return np.log(s)

    log_scales = _as_log_scales(default_scale)
    scales = np.tile(log_scales[None, :], (n, 1))
    scale_0, scale_1, scale_2 = scales[:, 0], scales[:, 1], scales[:, 2]

    # Colors -> f_dc_* (invert the transformation used by PlyLoader)
    per_vertex_rgb = _extract_vertex_rgb01(mesh)
    if per_vertex_rgb is None or len(per_vertex_rgb) != n:
        rgb01 = np.tile(np.asarray(default_rgb01, dtype=np.float32), (n, 1))
    else:
        rgb01 = per_vertex_rgb.astype(np.float32)
        # If sampled points don't match vertex count, we can't map per-vertex colors;
        # For sampled points, we fall back to default color.
        if rgb01.shape[0] != n:
            rgb01 = np.tile(np.asarray(default_rgb01, dtype=np.float32), (n, 1))

    fdc = _rgb_to_fdc(rgb01).astype(np.float32)
    f_dc_0, f_dc_1, f_dc_2 = fdc[:, 0], fdc[:, 1], fdc[:, 2]

    # 3) Build meshio object with only points + point_data
    # meshio expects float64 points; point_data arrays must be length-N
    points = pts.astype(np.float64)
    point_data = {
        "opacity": opacity_logit,
        "rot_0": rot_0,
        "rot_1": rot_1,
        "rot_2": rot_2,
        "rot_3": rot_3,
        "scale_0": scale_0,
        "scale_1": scale_1,
        "scale_2": scale_2,
        "f_dc_0": f_dc_0,
        "f_dc_1": f_dc_1,
        "f_dc_2": f_dc_2,
    }

    # Create a meshio "point cloud" (no cells)
    ply = meshio.Mesh(points=points, cells=[], point_data=point_data)

    # 4) Write as PLY; both file path and BytesIO are supported by meshio
    if isinstance(target, io.BytesIO):
        meshio.write(target, ply, file_format="ply")
        target.seek(0)
    else:
        meshio.write(target, ply, file_format="ply")
