from __future__ import annotations
from typing import Optional

from PIL import Image
from trimesh.exchange.ply import export_ply
import torch

from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline


class HunYuanImageTo3D:
    def __init__(self, device: torch.device):
        self.device = device

        self.dit_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2"
        )
        self.dit_pipeline.to(self.device)

        self.paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            "tencent/Hunyuan3D-2"
        )

    @torch.inference_mode()
    def infer_to_ply(self, image: Image.Image, seed: Optional[int] = None) -> bytes:
        if seed is not None:
            torch.manual_seed(seed)

        shape_mesh = self.dit_pipeline(image=image)[0]
        full_mesh = self.paint_pipeline(shape_mesh, image=image)

        return export_ply(full_mesh)
