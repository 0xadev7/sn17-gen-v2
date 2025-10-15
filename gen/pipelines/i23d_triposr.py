from __future__ import annotations

import numpy as np
from PIL import Image
import trimesh as tm
import torch

from tsr.system import TSR
from tsr.utils import resize_foreground
from gen.utils.mesh import mesh_to_binary_ply_bytes


class TripoSRImageTo3D:
    def __init__(self, device: str = "cuda", chunk_size: int = 8192):
        self.device = device

        self.pipe = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )

        self.pipe.renderer.set_chunk_size(chunk_size)
        self.pipe.to(device)

    def infer_to_ply(
        self,
        image: Image.Image,
        seed=None,
    ) -> tm.Trimesh:
        if seed is not None:
            torch.manual_seed(seed)

        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))

        scene_codes = self.pipe([image], device=self.device)
        meshes = self.pipe.extract_mesh(scene_codes, True)
        return mesh_to_binary_ply_bytes(meshes[0])
