import io, gc, struct
from contextlib import nullcontext
from typing import Optional, Tuple
import numpy as np
import torch
from PIL import Image

import trimesh as tm
from trimesh.sample import sample_surface


def _uint8_to_int8(u: np.ndarray) -> np.ndarray:
    # Map 0..255 -> -128..127 (centered). Matches what your logs showed (dtype=int8 with negatives).
    return (u.astype(np.int16) - 128).astype(np.int8)


def _compute_bbox_diag(points: np.ndarray) -> float:
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    return float(np.linalg.norm(mx - mn))


def _face_colors_from_vertices(mesh: tm.Trimesh) -> Optional[np.ndarray]:
    """
    Returns per-face RGB uint8 by averaging vertex colors if available, else None.
    """
    vc = None
    if (
        isinstance(mesh.visual, tm.visual.color.ColorVisuals)
        and mesh.visual.vertex_colors is not None
    ):
        vc = mesh.visual.vertex_colors  # (V, 4 or 3), usually RGBA
        if vc.shape[1] >= 3:
            vc = vc[:, :3].astype(np.uint8)
        else:
            vc = None
    if vc is None:
        return None

    # average each face's vertex colors
    faces = mesh.faces  # (F, 3)
    fc = vc[faces].mean(axis=1)
    # clamp and cast
    fc = np.clip(fc, 0, 255).astype(np.uint8)
    return fc  # (F, 3)


def sample_mesh_to_gaussians(
    mesh: tm.Trimesh,
    n_points: int = 20000,
    opacity_logit: float = 2.0,  # sigmoid(2.0) ≈ 0.88
    scale_ratio: float = 0.005,  # isotropic scale = scale_ratio * bbox_diag
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      xyz: (N,3) float32
      rgb_i8: (N,3) int8     (centered)
      opacity: (N,1) float32 (pre-sigmoid)
      scale: (N,3) float32
      rot: (N,4) float32 (quaternion w,x,y,z)
    """
    # 1) Sample points on surface
    pts, face_idx = sample_surface(mesh, n_points)  # pts (N,3) float64, face_idx (N,)
    xyz = pts.astype(np.float32)

    # 2) Colors
    fc = _face_colors_from_vertices(mesh)  # (F,3) uint8 or None
    if fc is not None:
        rgb_u8 = fc[face_idx]  # (N,3) uint8
    else:
        # fallback: light gray
        rgb_u8 = np.full((n_points, 3), 200, dtype=np.uint8)
    rgb_i8 = _uint8_to_int8(rgb_u8)  # (N,3) int8

    # 3) Opacity (logit)
    opacity = np.full((n_points, 1), opacity_logit, dtype=np.float32)

    # 4) Isotropic scale using bbox diagonal
    diag = _compute_bbox_diag(xyz)
    s = np.full((n_points, 3), scale_ratio * diag, dtype=np.float32)

    # 5) Identity rotation quaternion (w, x, y, z) = (1, 0, 0, 0)
    rot = np.zeros((n_points, 4), dtype=np.float32)
    rot[:, 0] = 1.0

    return xyz, rgb_i8, opacity, s, rot


def write_gs_ply_binary_le(
    xyz: np.ndarray,  # float32 (N,3)
    rgb_i8: np.ndarray,  # int8   (N,3)  fields: red, green, blue
    opacity: np.ndarray,  # float32 (N,1) field: opacity (pre-sigmoid)
    scale: np.ndarray,  # float32 (N,3) fields: scale_0..2
    rot: np.ndarray,  # float32 (N,4) fields: rot_0..3 (w,x,y,z)
) -> bytes:
    N = xyz.shape[0]
    # Header using field names your validator seems to expect (red/green/blue + opacity).
    # Extra fields (scale_*, rot_*) match common GS conventions and are harmless if unused.
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property char red\n"  # int8
        "property char green\n"  # int8
        "property char blue\n"  # int8
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    ).encode("ascii")

    buf = io.BytesIO()
    buf.write(header)

    # Pack each vertex record in little-endian order
    # < f f f b b b f f f f f f f f  (3 floats, 3 int8, 1 float, 3 floats, 4 floats)
    pack = struct.Struct("<fffbbbffffffff").pack

    # Ensure dtypes
    xyz = xyz.astype(np.float32, copy=False)
    rgb_i8 = rgb_i8.astype(np.int8, copy=False)
    opacity = opacity.astype(np.float32, copy=False)
    scale = scale.astype(np.float32, copy=False)
    rot = rot.astype(np.float32, copy=False)

    for i in range(N):
        x, y, z = xyz[i]
        r, g, b = rgb_i8[i]
        op = float(opacity[i, 0])
        s0, s1, s2 = scale[i]
        q0, q1, q2, q3 = rot[i]
        buf.write(pack(x, y, z, int(r), int(g), int(b), op, s0, s1, s2, q0, q1, q2, q3))

    return buf.getvalue()
