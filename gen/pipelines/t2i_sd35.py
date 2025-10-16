from __future__ import annotations
from typing import Optional
import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
import asyncio


class SD35Text2Image:
    def __init__(
        self,
        device: torch.device,
        model_id: str = "stabilityai/stable-diffusion-3.5-large-turbo",
    ):
        self.device = device
        self._lock = asyncio.Lock()  # 🔒 serialize SD3.5

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )

        self.pipe.to(self.device)

    async def generate_async(self, **kw):
        # call this from asyncio; wraps the sync generate
        async with self._lock:
            return await asyncio.to_thread(self._generate_sync, **kw)

    def _generate_sync(
        self,
        prompt,
        negative_prompt=None,
        res=768,
        steps=4,
        guidance=0.0,
        seed=None,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        width = (res // 16) * 16
        height = (res // 16) * 16

        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
        )
        return out.images[0]
