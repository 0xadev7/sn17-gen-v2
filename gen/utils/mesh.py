import io, gc
from contextlib import nullcontext
from typing import Optional, Tuple
import numpy as np
import torch
from PIL import Image
import trimesh as tm
from trimesh.sample import sample_surface

from plyfile import PlyData, PlyElement


def inverse_sigmoid(p: np.ndarray) -> np.ndarray:
    # logit(p) with clipping for numerical safety
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """
    q: (..., 4) as (w,x,y,z)
    returns: (..., 3, 3)
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # normalize to be safe
    norm = np.sqrt(w * w + x * x + y * y + z * z) + 1e-9
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    # rotation matrix
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.stack(
        [
            np.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], axis=-1),
            np.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], axis=-1),
            np.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], axis=-1),
        ],
        axis=-2,
    )
    return R


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """
    R: (..., 3, 3)
    returns q: (..., 4) (w,x,y,z)
    """
    t = np.trace(R, axis1=-2, axis2=-1)
    w = np.sqrt(np.maximum(0.0, 1.0 + t)) / 2.0
    x = np.sqrt(np.maximum(0.0, 1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2])) / 2.0
    y = np.sqrt(np.maximum(0.0, 1.0 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2])) / 2.0
    z = np.sqrt(np.maximum(0.0, 1.0 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2])) / 2.0

    x = np.copysign(x, R[..., 2, 1] - R[..., 1, 2])
    y = np.copysign(y, R[..., 0, 2] - R[..., 2, 0])
    z = np.copysign(z, R[..., 1, 0] - R[..., 0, 1])
    q = np.stack([w, x, y, z], axis=-1)
    # normalize
    q /= np.linalg.norm(q, axis=-1, keepdims=True) + 1e-9
    return q


def apply_transform_to_quats(q: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Rotate quaternions by a 3x3 matrix T: R' = T @ R
    q: (N,4), T: (3,3)
    """
    R = quat_to_mat(q)  # (N,3,3)
    R2 = T[None, ...] @ R  # left-multiply
    return mat_to_quat(R2)


# ----------------------------
# GS attribute synthesis
# ----------------------------


def _compute_bbox_diag(points: np.ndarray) -> float:
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    return float(np.linalg.norm(mx - mn))


def _face_colors_from_vertices(mesh: tm.Trimesh) -> Optional[np.ndarray]:
    if (
        isinstance(mesh.visual, tm.visual.color.ColorVisuals)
        and mesh.visual.vertex_colors is not None
    ):
        vc = mesh.visual.vertex_colors
        if vc.shape[1] >= 3:
            return vc[:, :3].astype(np.uint8)
    return None


def _rgb_u8_to_linear01(u8: np.ndarray) -> np.ndarray:
    # sRGB -> linear approx (good enough for DC)
    s = u8.astype(np.float32) / 255.0
    a = 0.055
    lin = np.where(s <= 0.04045, s / 12.92, ((s + a) / (1 + a)) ** 2.4)
    return lin


def sample_mesh_to_gs(
    mesh: tm.Trimesh,
    n_points: int = 20000,
    opacity_logit: float = 2.0,  # sigmoid(2) ~ 0.88 → stored as logit
    scale_ratio: float = 0.005,  # isotropic s = ratio * bbox_diag, then Trellis stores log(s)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns Trellis-ordered arrays:
      xyz       (N,3) float32
      normals   (N,3) float32 (zeros)
      f_dc      (N,3) float32 (DC SH, approx from linear RGB)
      opacity   (N,1) float32 (pre-sigmoid)
      scale_log (N,3) float32 (log of positive scale)
      rot       (N,4) float32 (w,x,y,z), identity
    """
    pts, face_idx = sample_surface(mesh, n_points)  # pts (N,3) float64
    xyz = pts.astype(np.float32)

    normals = np.zeros_like(xyz, dtype=np.float32)

    fc = _face_colors_from_vertices(mesh)
    if fc is None:
        fc = np.full((mesh.vertices.shape[0], 3), 200, dtype=np.uint8)  # fallback
    rgb_u8 = fc[face_idx]  # (N,3) uint8
    f_dc = _rgb_u8_to_linear01(rgb_u8).astype(np.float32)  # (N,3) float32

    opacity = np.full((n_points, 1), opacity_logit, dtype=np.float32)

    diag = _compute_bbox_diag(xyz)
    s = np.full((n_points, 3), max(1e-6, scale_ratio * diag), dtype=np.float32)
    scale_log = np.log(s)

    rot = np.zeros((n_points, 4), dtype=np.float32)
    rot[:, 0] = 1.0  # identity quaternions (w=1)

    return xyz, normals, f_dc, opacity, scale_log, rot


def write_gs_ply_bytes(
    xyz: np.ndarray,
    normals: np.ndarray,
    f_dc: np.ndarray,
    opacity: np.ndarray,
    scale_log: np.ndarray,
    rot: np.ndarray,
) -> bytes:
    """
    Write binary little-endian PLY with the exact Trellis field names & types:
      x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3
    All float32.
    """
    N = xyz.shape[0]
    attr_names = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    dtype_full = [(a, "f4") for a in attr_names]

    elements = np.empty(N, dtype=dtype_full)
    attributes = np.concatenate(
        [xyz, normals, f_dc, opacity, scale_log, rot], axis=1
    ).astype(np.float32, copy=False)

    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    buf = io.BytesIO()
    PlyData([el], text=False).write(buf)  # binary little-endian
    return buf.getvalue()
