from __future__ import annotations
from typing import List, Optional, Sequence
from contextlib import nullcontext
from diffusers import DiffusionPipeline
import math
import torch
from PIL import Image
from gen.utils.vram import vram_guard


class Zero123MV:
    """
    Image -> Multi-View generator using Zero123-XL family.

    Returns a list of RGB PIL images (we'll do background removal *after*).
    """

    def __init__(self, device: torch.device, res: int = 768):
        self.device = device
        self.res = int(res)
        self.dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        self.pipe = DiffusionPipeline.from_pretrained(
            "ashawkey/zero123-xl-diffusers",
            torch_dtype=self.dtype,
        ).to(self.device)

        # modest VRAM friendliness
        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            pass

        self.pipe.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def generate_views(
        self,
        source: Image.Image,
        num_views: int = 8,
        yaws_deg: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Args:
            source: single RGB PIL image
            num_views: how many camera yaws to sample around object
            yaws_deg: if given, explicit yaw angles (degrees). Otherwise uniform around 360°
        Returns:
            List of RGB PIL images at requested viewpoints
        """
        src_rgb = source.convert("RGB")
        W = H = (self.res // 8) * 8  # keep friendly size

        if not yaws_deg or len(yaws_deg) == 0:
            num = max(1, int(num_views))
            yaws_deg = [i * (360.0 / num) for i in range(num)]

        g = torch.Generator(device="cuda" if self.device.type == "cuda" else "cpu")
        if seed is not None:
            g.manual_seed(int(seed))

        images: List[Image.Image] = []
        with vram_guard():
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=self.dtype)
                if self.device.type == "cuda"
                else nullcontext()
            )
            try:
                for yaw in yaws_deg:
                    with autocast_ctx:
                        if self.pipe is None:
                            raise RuntimeError(f"Pipeline is not initialized")
                        out = self.pipe(
                            image=src_rgb,
                            num_inference_steps=20,
                            guidance_scale=3.5,
                            height=H,
                            width=W,
                            generator=g,
                            camera={"yaw": float(yaw), "pitch": 0.0, "radius": 1.5},
                        )
                    img = out.images[0]
                    images.append(img.copy())
                    try:
                        img.close()
                    except Exception:
                        pass
                    del out
            finally:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        return images
