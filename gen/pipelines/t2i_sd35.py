from __future__ import annotations
from typing import Optional
import torch
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from PIL import Image
import asyncio


class SD35Text2Image:
    def __init__(
        self,
        device: torch.device,
        model_id: str = "stabilityai/stable-diffusion-3.5-large",
        torch_dtype: torch.dtype = torch.float16,
        enable_xformers: bool = True,
        cpu_offload: bool = False,
    ):
        self.device = device
        self._lock = asyncio.Lock()  # 🔒 serialize SD3.5

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id, torch_dtype=torch_dtype
        )
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

        if enable_xformers and hasattr(
            self.pipe, "enable_xformers_memory_efficient_attention"
        ):
            self.pipe.enable_xformers_memory_efficient_attention()

        if cpu_offload and hasattr(self.pipe, "enable_model_cpu_offload"):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

        self.pipe.set_progress_bar_config(disable=True)

    async def generate_async(self, **kw):
        # call this from asyncio; wraps the sync generate
        async with self._lock:
            return await asyncio.to_thread(self._generate_sync, **kw)

    def _generate_sync(
        self,
        prompt,
        negative_prompt=None,
        res=768,
        steps=20,
        guidance=4.0,
        seed=None,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        self.pipe.scheduler.set_timesteps(steps)

        if hasattr(self.pipe.scheduler, "sigmas") and len(
            self.pipe.scheduler.timesteps
        ) == len(self.pipe.scheduler.sigmas):
            self.pipe.scheduler.timesteps = self.pipe.scheduler.timesteps[:-1]

        width = (res // 16) * 16
        height = (res // 16) * 16

        with torch.autocast("cuda" if self.device.type == "cuda" else "cpu"):
            out = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
            )
        return out.images[0]
