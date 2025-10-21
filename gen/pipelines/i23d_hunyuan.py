from __future__ import annotations
from typing import Optional

from PIL import Image
from trimesh.exchange.ply import export_ply
import torch
import sys

from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline


try:
    from utils.torchvision_fix import apply_fix

    apply_fix()
except ImportError:
    print(
        "Warning: torchvision_fix module not found, proceeding without compatibility fix"
    )
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")


class HunYuanImageTo3D:
    def __init__(self, device: torch.device):
        self.device = device

        self.shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2",
            subfolder="hunyuan3d-dit-v2-0-turbo"
        )

        self.shape_pipeline.to(self.device)

        self.paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            "tencent/Hunyuan3D-2"
        )

    @torch.inference_mode()
    def infer_to_ply(self, image: Image.Image, seed: Optional[int] = None) -> bytes:
        if seed is not None:
            torch.manual_seed(seed)

        shape_mesh = self.shape_pipeline(image=image)[0]
        full_mesh = self.paint_pipeline(shape_mesh, image=image)

        return export_ply(full_mesh)
