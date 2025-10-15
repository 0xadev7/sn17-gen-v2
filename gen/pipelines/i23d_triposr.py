from __future__ import annotations

import numpy as np
from PIL import Image
import trimesh as tm
import torch
from typing import Optional
from contextlib import nullcontext
import gc

from tsr.system import TSR
from gen.utils.mesh import (
    sample_mesh_to_gs,
    write_gs_ply_bytes,
    apply_transform_to_quats,
)


class TripoSRImageTo3D:
    def __init__(self, device: torch.device, chunk_size: int = 8192):
        self.device = device

        self.pipe = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )

        self.pipe.renderer.set_chunk_size(chunk_size)
        self.pipe.to(device)

    @torch.inference_mode()
    def infer_to_ply(
        self,
        image: Image.Image,
        seed: Optional[int] = None,
        n_points: int = 20000,
        opacity_logit: float = 2.0,
        scale_ratio: float = 0.005,
        transform: Optional[np.ndarray] = np.array(
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32
        ),  # Trellis default
    ) -> bytes:
        """
        TripoSR -> Mesh -> Trellis-style GS PLY bytes.

        Fields (all float32): x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3
        - opacity is pre-sigmoid (logit)
        - scale is log(s)
        - rotation is quaternion (w,x,y,z)
        """

        lock = getattr(self, "_lock", None)
        ctx = lock if lock is not None else nullcontext()
        with ctx:
            try:
                if seed is not None:
                    torch.manual_seed(seed)
                    np.random.seed(seed & 0xFFFFFFFF)

                # RGB input
                if image.mode != "RGB":
                    image = image.convert("RGB")

                if self.device.type == "cuda":
                    torch.cuda.set_device(self.device)
                    torch.cuda.synchronize(self.device)

                # ----- TripoSR forward -> mesh -----
                scene_codes = self.pipe([image], device=self.device)
                meshes = self.pipe.extract_mesh(scene_codes, True)
                mesh: tm.Trimesh = meshes[0]

                # ----- Mesh -> Trellis GS attributes -----
                xyz, normals, f_dc, opacity, scale_log, rot = sample_mesh_to_gs(
                    mesh,
                    n_points=n_points,
                    opacity_logit=opacity_logit,
                    scale_ratio=scale_ratio,
                )

                # ----- Optional transform (same semantics as Trellis) -----
                if transform is not None:
                    T = np.asarray(transform, dtype=np.float32)
                    xyz = (xyz @ T.T).astype(np.float32, copy=False)
                    # rotate quaternions by T
                    rot = apply_transform_to_quats(rot, T).astype(
                        np.float32, copy=False
                    )

                # ----- Write Trellis-style PLY -----
                ply_bytes = write_gs_ply_bytes(
                    xyz, normals, f_dc, opacity, scale_log, rot
                )

                # Cleanup
                del (
                    scene_codes,
                    meshes,
                    mesh,
                    xyz,
                    normals,
                    f_dc,
                    opacity,
                    scale_log,
                    rot,
                )
                gc.collect()
                if self.device.type == "cuda":
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize(self.device)

                return ply_bytes

            except RuntimeError as e:
                msg = str(e).lower()
                if (
                    ("illegal memory access" in msg)
                    or ("device-side assert" in msg)
                    or ("out of memory" in msg)
                ):
                    self._poisoned = True
                    if self.device.type == "cuda":
                        with torch.cuda.device(self.device):
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                raise
