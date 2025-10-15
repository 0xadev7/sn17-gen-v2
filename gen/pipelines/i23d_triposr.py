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
from gen.utils.mesh import mesh_to_binary_ply_bytes


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
    ) -> bytes:
        """
        Synchronous and device-guarded. Mirrors Trellis' infer_to_ply contract.
        Returns: binary little-endian PLY bytes.
        """
        lock = getattr(self, "_lock", None)
        ctx = lock if lock is not None else nullcontext()

        with ctx:
            try:
                if seed is not None:
                    torch.manual_seed(seed)
                    np.random.seed(seed & 0xFFFFFFFF)

                # Device guard & sync
                if self.device.type == "cuda":
                    torch.cuda.set_device(self.device)
                    torch.cuda.synchronize(self.device)

                image = np.array(image).astype(np.float32) / 255.0
                image = (
                    image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
                )
                image = Image.fromarray((image * 255.0).astype(np.uint8))

                # --- TripoSR forward ---
                # Most TripoSR wrappers accept a list of PIL Images and a device kwarg.
                scene_codes = self.pipe([image], device=self.device)
                # `to_trimesh=True` (or the boolean flag in your wrapper) should return trimesh objects.
                meshes = self.pipe.extract_mesh(scene_codes, True)
                mesh = meshes[0]

                # Convert to binary PLY bytes (match Trellis)
                try:
                    ply_bytes = mesh_to_binary_ply_bytes(mesh)
                finally:
                    # Cleanup intermediates before CUDA cache clear
                    del scene_codes, meshes, mesh

                # Post-run sync & memory hygiene
                gc.collect()
                if self.device.type == "cuda":
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize(self.device)

                return ply_bytes

            except RuntimeError as e:
                # Treat OOM/illegal access as "poison" to force a clean re-init on next call,
                # exactly like the Trellis example does.
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
