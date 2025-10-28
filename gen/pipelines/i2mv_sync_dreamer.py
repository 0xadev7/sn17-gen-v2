from __future__ import annotations
from typing import List, Optional, Dict, Any
import os
import tempfile
import numpy as np
import torch
from PIL import Image

from omegaconf import OmegaConf

# SyncDreamer modules (use your vendored path)
from ldm.util import instantiate_from_config, prepare_inputs
from ldm.models.diffusion.sync_dreamer import (
    SyncMultiviewDiffusion,
    SyncDDIMSampler,
)

from gen.utils.vram import vram_guard


class SyncDreamerMV:
    """
    Image -> Multi-View generator using SyncDreamer (official repo APIs).

    Notes
    -----
    - This adapter loads your local config/ckpt and exposes a simple API returning
      a list of RGB PIL images (one per view).
    - It follows the official generator.py logic:
        * prepare_inputs(image_path, elevation, crop_size)
        * sample with SyncDDIMSampler
        * decode to 0..255 uint8
    - We keep sample_num=1 by default, since our downstream Trellis consumes
      a single MV set (N views) per object.
    """

    def __init__(
        self,
        device: torch.device,
        cfg_path: str = "gen/lib/sync_dreamer/configs/syncdreamer.yaml",
        ckpt_path: str = "gen/lib/sync_dreamer/ckpt/syncdreamer.ckpt",
        strict_ckpt: bool = True,
        *,
        res: int = 768,
        elevation: float = 0.0,
        sample_steps: int = 50,
        cfg_scale: float = 2.0,
    ):
        self.device = device
        self.res = int(res)

        # Load model (same as generator.py: load_model)
        config = OmegaConf.load(cfg_path)
        model = instantiate_from_config(config.model)
        if not isinstance(model, SyncMultiviewDiffusion):
            raise RuntimeError(
                f"Loaded model is not SyncMultiviewDiffusion. Got: {type(model)}"
            )

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=strict_ckpt)
        self.model: SyncMultiviewDiffusion = model.to(self.device).eval()

        # Default sampler is DDIM (same as generator.py)
        self.default_sampler = "ddim"

        self.elevation = elevation
        self.sample_steps = sample_steps
        self.cfg_scale = cfg_scale

    @torch.inference_mode()
    def generate_views(
        self,
        source: Image.Image,
        num_views: int = 8,
        *,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Parameters
        ----------
        source : PIL.Image
            Input RGB image of the object.
        num_views : int
            How many viewpoints to synthesize (N).
        seed : int or None
            RNG seed for reproducibility.

        Returns
        -------
        List[PIL.Image]
            A list of N RGB images, one per view. If sample_num>1, we return the
            first sample’s views (consistent with our downstream Trellis usage).
        """
        # SyncDreamer’s utilities expect a path; write a temp file then call prepare_inputs.
        # This avoids re-implementing internal preprocessing.
        tmp_dir = None
        tmp_path = None
        sample_num = num_views

        # Normalize seed
        if seed is not None:
            seed = int(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Prepare inputs dictionary using official helper
        try:
            # Save input image to temp file
            tmp_dir = tempfile.TemporaryDirectory(prefix="syncdreamer_")
            tmp_path = os.path.join(tmp_dir.name, "input.png")
            src_rgb = source.convert("RGB")
            # Optionally resize to a friendly target; SyncDreamer handles its own internal sizing.
            # We keep original to preserve identity.
            src_rgb.save(tmp_path)

            # prepare_inputs returns tensors on CPU; we move to CUDA and repeat for sample_num
            data: Dict[str, Any] = prepare_inputs(
                tmp_path, self.elevation, crop_size=-1, image_size=self.res
            )
            for k, v in data.items():
                v = v.unsqueeze(0)  # [1, ...]
                v = torch.repeat_interleave(v, int(sample_num), dim=0)  # [B, ...]
                data[k] = v.to(self.device, non_blocking=True)

            # Build sampler (DDIM only per current generator.py)
            if self.default_sampler != "ddim":
                raise NotImplementedError("Only DDIM sampler is supported for now.")
            sampler = SyncDDIMSampler(self.model, ddim_num_steps=int(self.sample_steps))

            # Run sampling
            with vram_guard():
                x_sample = self.model.sample(sampler, data, float(self.cfg_scale), 8)

            # x_sample: [B, N, C, H, W], with values in [-1, 1]
            if x_sample.ndim != 5:
                raise RuntimeError(f"Unexpected sample shape: {tuple(x_sample.shape)}")

            B, N, C, H, W = x_sample.shape
            # If the model produced more views than requested, truncate; if fewer, keep all.
            N = min(N, int(num_views))
            x_sample = x_sample[:, :N]

            # Convert to uint8 images (0..255), first sample only
            x_sample = (torch.clamp(x_sample, min=-1.0, max=1.0) + 1) * 0.5
            x_sample = x_sample.permute(0, 1, 3, 4, 2).contiguous()  # [B, N, H, W, C]
            arr = (x_sample.detach().cpu().numpy() * 255).astype(np.uint8)

            # We return the first batch’s N views as a list of PIL images
            out: List[Image.Image] = []
            for ni in range(N):
                out.append(Image.fromarray(arr[0, ni], mode="RGB"))

            return out

        finally:
            # Cleanup temp artifacts
            try:
                if tmp_dir is not None:
                    tmp_dir.cleanup()
            except Exception:
                pass
