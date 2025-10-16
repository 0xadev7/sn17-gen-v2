from __future__ import annotations

import numpy as np
from PIL import Image
import trimesh as tm
import torch
from typing import Optional
from contextlib import nullcontext
import gc
import io
import os

from tsr.system import TSR
from gen.utils.ply_writer import triposr_meshes_to_gs_ply_bytes
from tsr.utils import save_video


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
        self, image: Image.Image, seed: Optional[int] = None, debug_save=False
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

                print('extracted meshes')
                # mesh + texture -> PLY
                ply_bytes = triposr_meshes_to_gs_ply_bytes(
                    meshes,
                    model=self.pipe,  # the TripoSR model you used above
                    scene_code=scene_codes[0],
                    n_samples=20000,
                    texture_resolution=2048,
                    opacity=0.9,
                    scale_mult=0.5,  # tweak if splats feel too big/small
                )

                # Cleanup
                del (
                    scene_codes,
                    meshes,
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
