from __future__ import annotations

import numpy as np
from PIL import Image
import trimesh as tm
import torch
from typing import Optional
from contextlib import nullcontext
import gc

from tsr.system import TSR
from tsr.utils import resize_foreground
from gen.utils.mesh import sample_mesh_to_gaussians, write_gs_ply_binary_le


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
    ) -> bytes:
        """
        TripoSR -> Mesh -> Gaussian Splat PLY (Trellis-style).
        Returns binary little-endian PLY bytes with fields:
        x y z red green blue opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3
        """
        lock = getattr(self, "_lock", None)
        ctx = lock if lock is not None else nullcontext()
        with ctx:
            try:
                if seed is not None:
                    torch.manual_seed(seed)
                    np.random.seed(seed & 0xFFFFFFFF)

                # RGB input (TripoSR expects RGB)
                if image.mode != "RGB":
                    image = image.convert("RGB")

                # Device guards
                if self.device.type == "cuda":
                    torch.cuda.set_device(self.device)
                    torch.cuda.synchronize(self.device)

                # ----- TripoSR forward -> mesh -----
                scene_codes = self.pipe([image], device=self.device)
                meshes = self.pipe.extract_mesh(scene_codes, True)
                mesh: tm.Trimesh = meshes[0]

                # ----- Convert mesh -> GS attributes -----
                xyz, rgb_i8, opacity, scale, rot = sample_mesh_to_gaussians(
                    mesh,
                    n_points=n_points,
                    opacity_logit=opacity_logit,
                    scale_ratio=scale_ratio,
                )

                # ----- Write Trellis-style GS PLY -----
                ply_bytes = write_gs_ply_binary_le(xyz, rgb_i8, opacity, scale, rot)

                # Cleanup
                del scene_codes, meshes, mesh, xyz, rgb_i8, opacity, scale, rot
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
