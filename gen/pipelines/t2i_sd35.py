from __future__ import annotations
from typing import Optional
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image


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
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=torch_dtype
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

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        res: int = 768,
        steps: int = 20,
        guidance: float = 4.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        if seed is not None:
            torch.manual_seed(seed)

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=res,
            height=res,
        )
        return out.images[0]
