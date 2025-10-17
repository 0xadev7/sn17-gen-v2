from __future__ import annotations
from typing import Optional

from PIL import Image
from trimesh.exchange.ply import export_ply
import torch

from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dshape.hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline


class HunYuanImageTo3D:
    def __init__(self, device: torch.device):
        self.device = device

        self.shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2.1"
        )

        self.shape_pipeline.to(self.device)

        self.paint_pipeline = Hunyuan3DPaintPipeline(
            Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
        )

    @torch.inference_mode()
    def infer_to_ply(self, image: Image.Image, seed: Optional[int] = None) -> bytes:
        if seed is not None:
            torch.manual_seed(seed)

        shape_mesh = self.shape_pipeline(image=image)[0]
        full_mesh = self.paint_pipeline(shape_mesh, image=image)

        return export_ply(full_mesh)
