from __future__ import annotations
from typing import Optional
import numpy as np
from PIL import Image
import trimesh as tm
from tsr.system import TSR

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

    def infer_to_ply(self, image: Image.Image) -> tm.Trimesh:
        scene_codes = self.pipe([image], device=self.device)
        meshes = self.pipe.extract_mesh(scene_codes)
        return mesh_to_binary_ply_bytes(meshes[0])
