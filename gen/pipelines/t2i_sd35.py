from __future__ import annotations
import torch
from diffusers import StableDiffusion3Pipeline


class SD35Text2Image:
    def __init__(
        self,
        device: torch.device,
    ):
        self.device = device

        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large-turbo", torch_dtype=torch.bfloat16
        )
        self.pipeline.to(self.device)

    def generate(
        self,
        prompt,
        steps=4,
        res=1024,
        seed=None,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        width = (res // 16) * 16
        height = (res // 16) * 16

        out = self.pipeline(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=0.0,
            width=width,
            height=height,
        )
        return out.images[0]
