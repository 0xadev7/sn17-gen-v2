import io
import math
import numpy as np
from PIL import Image
import meshio
import trimesh as tm
from tsr.bake_texture import bake_texture

# If you already have TripoSR's bake_texture import, use that.
# from triposr.xatlas_utils import bake_texture  # example; adjust to your path
# In your sample you call: bake_texture(mesh, model, scene_code, texture_resolution)

_SH_C0 = 0.28209479177387814
_MEAN = 0.5


def _logit(p: float) -> float:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return float(np.log(p / (1.0 - p)))


def _unit_quat() -> np.ndarray:
    # Identity rotation (w, x, y, z)
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _estimate_isotropic_scale(
    V: np.ndarray, F: np.ndarray, scale_mult: float = 0.5
) -> float:
    """
    Estimate a reasonable splat radius from mesh geometry.
    Uses median edge length (robust to outliers), then scales.
    """
    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    unique_edges = np.unique(np.sort(edges, axis=1), axis=0)
    e0, e1 = V[unique_edges[:, 0]], V[unique_edges[:, 1]]
    elens = np.linalg.norm(e1 - e0, axis=1)
    med = np.median(elens) if elens.size else 1.0
    return float(max(med * scale_mult, 1e-4))


def _rand_barycentric(n: int) -> np.ndarray:
    """
    Area-uniform random barycentric coordinates on triangles.
    """
    u = np.random.rand(n).astype(np.float32)
    v = np.random.rand(n).astype(np.float32)
    s = np.sqrt(u)
    b0 = 1.0 - s
    b1 = s * (1.0 - v)
    b2 = s * v
    return np.stack([b0, b1, b2], axis=1)  # (n,3)


def _sample_points_with_uv(
    V: np.ndarray, F: np.ndarray, UV: np.ndarray, n_samples: int
):
    """
    Sample 'n_samples' surface points with corresponding interpolated UVs.

    V  : (Nu,3) vertex positions in *UV-split* vertex space (vmapped)
    F  : (T,3)  faces indexing into V and UV
    UV : (Nu,2) vertex UVs aligned with V after vmapping
    """
    # Face areas to draw triangles proportionally to area
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) * 0.5
    areas = areas.astype(np.float64)
    probs = areas / (areas.sum() + 1e-12)

    # Choose faces
    face_idx = np.random.choice(len(F), size=n_samples, p=probs)
    bc = _rand_barycentric(n_samples)

    f = F[face_idx]
    p = (
        V[f[:, 0]] * bc[:, [0]] + V[f[:, 1]] * bc[:, [1]] + V[f[:, 2]] * bc[:, [2]]
    )  # (n,3)

    uv = (
        UV[f[:, 0]] * bc[:, [0]] + UV[f[:, 1]] * bc[:, [1]] + UV[f[:, 2]] * bc[:, [2]]
    )  # (n,2)

    return p.astype(np.float32), uv.astype(np.float32), face_idx


def _sample_colors_from_texture(
    texture_rgb01: np.ndarray, uv: np.ndarray
) -> np.ndarray:
    """
    Bilinear sample colors in [0,1] from texture at given UVs in [0,1].
    texture_rgb01: (H,W,3) float32 in [0,1]
    uv: (n,2) with uv[:,0]=U (x), uv[:,1]=V (y). Assumes V-up in baking; flip if needed.
    """
    H, W, _ = texture_rgb01.shape

    # Many UV unwraps use V pointing up; PIL save in sample flips vertically.
    # The TripoSR sample does: Image(...).transpose(Image.FLIP_TOP_BOTTOM)
    # We therefore flip V here to match that export.
    u = np.clip(uv[:, 0], 0.0, 1.0) * (W - 1)
    v = (1.0 - np.clip(uv[:, 1], 0.0, 1.0)) * (H - 1)

    x0 = np.floor(u).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.floor(v).astype(int)
    y1 = np.clip(y0 + 1, 0, H - 1)

    wx = (u - x0).reshape(-1, 1)
    wy = (v - y0).reshape(-1, 1)

    c00 = texture_rgb01[y0, x0]
    c10 = texture_rgb01[y0, x1]
    c01 = texture_rgb01[y1, x0]
    c11 = texture_rgb01[y1, x1]

    c0 = c00 * (1 - wx) + c10 * wx
    c1 = c01 * (1 - wx) + c11 * wx
    c = c0 * (1 - wy) + c1 * wy
    return np.clip(c, 0.0, 1.0).astype(np.float32)


def _colors_to_f_dc(rgb01: np.ndarray) -> np.ndarray:
    """
    Convert RGB in [0,1] to stored DC coefficients so that:
      features_dc = 0.5 + 0.28209479 * f_dc  ≈ RGB
    """
    fdc = (rgb01 - _MEAN) / _SH_C0
    return fdc.astype(np.float32)


def _build_point_data(
    points: np.ndarray, rgb01: np.ndarray, log_scale: float, opacity_logit: float
):
    n = points.shape[0]
    rot = np.tile(_unit_quat(), (n, 1))  # (n,4)
    scales = np.full((n, 3), log_scale, dtype=np.float32)  # isotropic in log-space
    fdc = _colors_to_f_dc(rgb01)  # (n,3)

    # Meshio point_data wants dict of arrays with shape (n,)
    point_data = {
        "opacity": np.full((n,), opacity_logit, dtype=np.float32),
        "rot_0": rot[:, 0].astype(np.float32),
        "rot_1": rot[:, 1].astype(np.float32),
        "rot_2": rot[:, 2].astype(np.float32),
        "rot_3": rot[:, 3].astype(np.float32),
        "scale_0": scales[:, 0],
        "scale_1": scales[:, 1],
        "scale_2": scales[:, 2],
        "f_dc_0": fdc[:, 0],
        "f_dc_1": fdc[:, 1],
        "f_dc_2": fdc[:, 2],
    }
    return point_data


def bake_triposr_texture(
    mesh: tm.Trimesh, model, scene_code, texture_resolution: int = 2048
):
    """
    Runs TripoSR texture baking and returns UV-split geometry + texture image (float [0,1]).
    Expected keys of the returned dict:
      V_uvsplit: (Nu,3) vertices aligned with UVs (mesh.vertices[vmapping])
      F:         (T,3) face indices into V_uvsplit / UV
      UV:        (Nu,2) uv coords aligned with V_uvsplit
      tex:       (H,W,3) float32 in [0,1]
    """
    # Your project already has this symbol available per the sample.
    print("baking texture")
    bake_output = bake_texture(mesh, model, scene_code, texture_resolution)
    print("baked texture")

    V_uvsplit = mesh.vertices[bake_output["vmapping"]]
    F = bake_output["indices"].astype(np.int32)
    UV = bake_output["uvs"].astype(np.float32)

    # TripoSR sample flips vertically when saving; keep float [0,1] here.
    tex01 = np.clip(bake_output["colors"], 0.0, 1.0).astype(np.float32)

    return dict(V_uvsplit=V_uvsplit, F=F, UV=UV, tex=tex01)


def generate_gaussian_ply_bytes_from_bake(
    bake_dict: dict,
    n_samples: int = 20000,
    opacity: float = 0.9,
    scale_mult: float = 0.5,
) -> bytes:
    """
    Converts a baked TripoSR mesh into Gaussian-splatting PLY bytes
    that your PlyLoader can consume.
    """
    V = bake_dict["V_uvsplit"]
    F = bake_dict["F"]
    UV = bake_dict["UV"]
    tex01 = bake_dict["tex"]

    # 1) sample points + UVs
    pts, uvs, _ = _sample_points_with_uv(V, F, UV, n_samples)
    print("mesh -> ply: step1 finished")

    # 2) colors from texture
    rgb = _sample_colors_from_texture(tex01, uvs)
    print("mesh -> ply: step2 finished")

    # 3) log-space scale and per-point attributes
    splat_scale = _estimate_isotropic_scale(V, F, scale_mult=scale_mult)
    log_s = float(np.log(max(splat_scale, 1e-6)))
    op_logit = _logit(opacity)

    point_data = _build_point_data(pts, rgb, log_s, op_logit)
    
    print("mesh -> ply: step3 finished")

    # 4) write PLY point cloud
    ply_mesh = meshio.Mesh(points=pts, cells=[])  # point cloud; no faces needed
    buf = io.BytesIO()
    meshio.write(buf, ply_mesh, file_format="ply", point_data=point_data)
    return buf.getvalue()


def triposr_meshes_to_gs_ply_bytes(
    meshes,
    model,
    scene_code,
    n_samples: int = 20000,
    texture_resolution: int = 2048,
    opacity: float = 0.9,
    scale_mult: float = 0.5,
) -> bytes:
    """
    End-to-end convenience: bake TripoSR texture on the first mesh
    and return Gaussian-splatting PLY bytes.
    """
    mesh: tm.Trimesh = meshes[0]
    bake_dict = bake_triposr_texture(mesh, model, scene_code, texture_resolution)
    return generate_gaussian_ply_bytes_from_bake(
        bake_dict,
        n_samples=n_samples,
        opacity=opacity,
        scale_mult=scale_mult,
    )
