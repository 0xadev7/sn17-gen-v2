from __future__ import annotations
import io, torch
from typing import Optional, List, Sequence, Any, Dict
from contextlib import nullcontext
from PIL import Image

from gen.lib.trellis.pipelines import TrellisImageTo3DPipeline
from gen.utils.vram import vram_guard


class TrellisImageTo3D:
    def __init__(self, device: torch.device):
        self.device = device
        self.dtype = torch.float16 if device.type == "cuda" else torch.float32

        self.pipe = TrellisImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS-image-large",
        )
        self.pipe.to(self.device)

    @torch.inference_mode()
    def infer_to_ply(
        self,
        image: Image.Image,
        struct_steps: Optional[int] = 8,
        slat_steps: Optional[int] = 12,
        cfg_struct: Optional[float] = 7.5,
        cfg_slat: Optional[float] = 3.0,
        seed: Optional[int] = None,
    ) -> bytes:
        """
        Single-view convenience wrapper (kept for backward compatibility).
        """
        return self.infer_multiview_to_ply(
            images=[image],
            struct_steps=struct_steps,
            slat_steps=slat_steps,
            cfg_struct=cfg_struct,
            cfg_slat=cfg_slat,
            seed=seed,
        )

    @torch.inference_mode()
    def infer_multiview_to_ply(
        self,
        images: Sequence[Image.Image],
        struct_steps: Optional[int] = 8,
        slat_steps: Optional[int] = 12,
        cfg_struct: Optional[float] = 7.5,
        cfg_slat: Optional[float] = 3.0,
        seed: Optional[int] = None,
    ) -> bytes:
        """
        Feed multiple views to Trellis in one go to reconstruct ONE object.
        Accepts a non-empty list/sequence of PIL images (RGB or RGBA).
        """
        if not images:
            return b""

        with vram_guard():
            outputs = None
            gs = None
            buf = None
            try:
                kwargs: Dict[str, Any] = dict(
                    seed=(seed if seed is not None else 1),
                    sparse_structure_sampler_params={
                        "steps": struct_steps,
                        "cfg_strength": cfg_struct,
                    },
                    slat_sampler_params={
                        "steps": slat_steps,
                        "cfg_strength": cfg_slat,
                    },
                )
                try:
                    outputs = self.pipe.run_multi_image(images=list(images), **kwargs)
                except TypeError:
                    # Fallback: use single image
                    outputs = self.pipe.run(image=images[0], **kwargs)

                gs = outputs["gaussian"][0]
                buf = io.BytesIO()
                gs.save_ply(buf)
                return buf.getvalue()

            finally:
                try:
                    if buf is not None:
                        buf.close()
                except Exception:
                    pass
                try:
                    del gs
                except Exception:
                    pass
                try:
                    del outputs
                except Exception:
                    pass

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
