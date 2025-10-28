from __future__ import annotations
from typing import List, Optional
from contextlib import nullcontext
import random
import torch
from PIL import Image

from gen.utils.vram import vram_guard
from gen.utils.prompt import tune_prompt

VIEW_LABELS = [
    "front view",
    "left side view",
    "right side view",
    "back view",
    "front three-quarter view",
    "rear three-quarter view",
    "top-down view",
    "low-angle front view",
]


class SD35Multiview:
    """
    Multiview generator using SD3.5:
      • Prefers StableDiffusion3Img2ImgPipeline when available (image -> image).
      • Falls back to StableDiffusion3Pipeline (text -> image) if img2img is missing.
    """

    def __init__(self, device: torch.device, res: int = 768):
        self.device = device
        self.res = int(res)
        self.dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        self.pipe_i2i = None
        self.pipe_t2i = None

        # Try img2img first
        try:
            from diffusers import StableDiffusion3Img2ImgPipeline

            self.pipe_i2i = StableDiffusion3Img2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-3.5-large-turbo",
                torch_dtype=self.dtype,
            ).to(self.device)
            self._post_init(self.pipe_i2i)
        except Exception:
            self.pipe_i2i = None

        # Always have a T2I fallback
        if self.pipe_i2i is None:
            from diffusers import StableDiffusion3Pipeline

            self.pipe_t2i = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3.5-large-turbo",
                torch_dtype=self.dtype,
            ).to(self.device)
            self._post_init(self.pipe_t2i)

    def _post_init(self, pipe):
        if self.device.type == "cuda":
            try:
                pipe.enable_vae_tiling()
            except Exception:
                pass
            try:
                pipe.enable_attention_slicing("max")
            except Exception:
                pass
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

    @torch.inference_mode()
    def generate_views(
        self,
        source: Image.Image,
        num_views: int = 8,
        base_prompt: Optional[str] = None,
        guidance: float = 2.0,
        steps: int = 14,
        strength: float = 0.8,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Args:
          source: base RGB image to condition on (used by img2img; for T2I fallback we still use it as a visual reference only)
          num_views: how many different viewpoint prompts to synthesize
          base_prompt: (optional) text prompt for consistency; pass your original text prompt when available
          guidance/steps/strength: SD knobs
        Returns:
          List of RGB PIL images
        """
        labels = VIEW_LABELS[: max(1, int(num_views))]
        W = H = (int(self.res) // 16) * 16

        g = torch.Generator(device="cuda" if self.device.type == "cuda" else "cpu")
        if seed is not None:
            g.manual_seed(int(seed))

        images: List[Image.Image] = []
        src_rgb = source.convert("RGB")

        # Compose a stable prompt
        base = (
            base_prompt
            or "the same object, consistent identity, plain background, studio lighting, high detail"
        )
        base = tune_prompt(base)

        with vram_guard():
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=self.dtype)
                if self.device.type == "cuda"
                else nullcontext()
            )
            try:
                for i, view in enumerate(labels):
                    prompt = f"{base}, {view}"
                    with autocast_ctx:
                        if self.pipe_i2i is not None:
                            out = self.pipe_i2i(
                                prompt=prompt,
                                image=src_rgb,
                                strength=float(strength),
                                guidance_scale=float(guidance),
                                num_inference_steps=int(steps),
                                width=W,
                                height=H,
                                generator=g,
                            )
                        else:
                            # T2I fallback
                            out = self.pipe_t2i(
                                prompt=prompt,
                                num_inference_steps=int(steps),
                                guidance_scale=float(guidance),
                                width=W,
                                height=H,
                                generator=g,
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
