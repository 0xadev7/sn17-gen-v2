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


def _triangulated_copy(mesh: tm.Trimesh) -> tm.Trimesh:
    """
    Create a triangle-only mesh copy that preserves vertices/faces and visuals as best as possible.
    """
    # as_triangles() returns a new Trimesh guaranteed to be triangular
    tri = mesh.as_triangles()
    # Ensure arrays are concrete numpy (not TrackedArray views) to avoid shape surprises
    tri.vertices = np.asarray(tri.vertices)
    tri.faces = np.asarray(tri.faces, dtype=np.int64)
    return tri


def _per_face_colors(tri: tm.Trimesh) -> np.ndarray:
    """
    Return (F,3) uint8 colors aligned with tri.faces. Prefer vertex_colors -> per-face mean.
    Fallback to face_colors if present and aligned. Else constant gray.
    """
    F = int(len(tri.faces))
    # default fallback
    gray = np.full((F, 3), 200, dtype=np.uint8)

    # Try vertex colors -> per-face average
    vc = None
    try:
        if (
            hasattr(tri.visual, "vertex_colors")
            and tri.visual.vertex_colors is not None
        ):
            vc = np.asarray(tri.visual.vertex_colors)
            if vc.ndim == 2 and vc.shape[0] == len(tri.vertices):
                if vc.shape[1] >= 3:
                    vc = vc[:, :3].astype(np.uint8)
                else:
                    vc = None
            else:
                vc = None
    except Exception:
        vc = None

    if vc is not None:
        faces = np.asarray(tri.faces, dtype=np.int64)
        # mean per face over its 3 vertices
        fc = vc[faces].mean(axis=1)
        fc = np.clip(fc, 0, 255).astype(np.uint8)
        if fc.shape[0] == F:
            return fc

    # Try face_colors aligned to faces
    try:
        if hasattr(tri.visual, "face_colors") and tri.visual.face_colors is not None:
            fc = np.asarray(tri.visual.face_colors)
            if fc.ndim == 2 and fc.shape[0] >= F and fc.shape[1] >= 3:
                return fc[:F, :3].astype(np.uint8)
    except Exception:
        pass

    return gray


def sample_mesh_to_gs(
    mesh: tm.Trimesh,
    n_points: int,
    opacity_logit: float,
    scale_ratio: float,
):
    """
    Robust sampler that guarantees color-face alignment and avoids IndexError.
    Returns:
      xyz (N,3) float32
      normals (N,3) float32 zeros
      f_dc (N,3) float32 (approx from linear RGB)
      opacity (N,1) float32 (logit)
      scale_log (N,3) float32
      rot (N,4) float32 identity quats
    """
    # 1) Work on a triangle-only mesh for both sampling and color extraction
    tri = _triangulated_copy(mesh)

    # 2) Sample points and face indices
    pts, face_idx = sample_surface(tri, n_points)
    pts = np.asarray(pts, dtype=np.float32)
    face_idx = np.asarray(face_idx, dtype=np.int64)

    # 3) Colors aligned to tri.faces
    fc = _per_face_colors(tri)  # (F,3) uint8 aligned with tri.faces

    F = fc.shape[0]
    # Safety: if any face_idx is unexpectedly out-of-range, clip to avoid crashes.
    if face_idx.max(initial=0) >= F or face_idx.min(initial=0) < 0:
        # You likely have a mismatch upstream; log and clip to proceed.
        # (Replace with your logger if available.)
        try:
            import logging

            logging.getLogger(__name__).warning(
                "sample_mesh_to_gs: face_idx out of bounds (max=%d, F=%d). Clipping.",
                int(face_idx.max(initial=-1)),
                F,
            )
        except Exception:
            pass
        face_idx = np.clip(face_idx, 0, F - 1)

    rgb_u8 = fc[face_idx]  # (N,3) uint8

    # 4) Normals: Trellis stores zeros
    normals = np.zeros_like(pts, dtype=np.float32)

    # 5) Approximate f_dc from linearized sRGB
    s = rgb_u8.astype(np.float32) / 255.0
    a = 0.055
    f_dc = np.where(s <= 0.04045, s / 12.92, ((s + a) / (1 + a)) ** 2.4).astype(
        np.float32
    )

    # 6) Opacity (logit)
    opacity = np.full((n_points, 1), float(opacity_logit), dtype=np.float32)

    # 7) Scale: log of isotropic radius from bbox diagonal
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    diag = float(np.linalg.norm(mx - mn))
    s_iso = max(1e-6, scale_ratio * diag)
    scale_log = np.log(np.full((n_points, 3), s_iso, dtype=np.float32))

    # 8) Rotation: identity quaternions
    rot = np.zeros((n_points, 4), dtype=np.float32)
    rot[:, 0] = 1.0

    return pts, normals, f_dc, opacity, scale_log, rot


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
