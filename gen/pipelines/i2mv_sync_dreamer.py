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
    - Loads local config/ckpt and returns a list of RGB PIL images (one per view).
    - Mirrors official generator.py:
        * prepare_inputs(image_path, elevation, crop_size, image_size)
        * sample with SyncDDIMSampler
        * decode to 0..255 uint8
    - sample_num=1 by default—downstream Trellis consumes one MV set.
    """

    def __init__(
        self,
        device: torch.device,
        cfg_path: str = "gen/lib/sync_dreamer/configs/syncdreamer.yaml",
        ckpt_path: str = "gen/lib/sync_dreamer/ckpt/syncdreamer.ckpt",
        strict_ckpt: bool = True,
        *,
        # Use a stable, square training size (SyncDreamer commonly uses 512)
        res: int = 512,
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
            A list of N RGB images, one per view.
        """
        tmp_dir = None
        sample_num = 1  # keep one MV set

        # Normalize seed
        if seed is not None:
            seed = int(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

        try:
            # Save input image to temp file (official utils expect a path)
            tmp_dir = tempfile.TemporaryDirectory(prefix="syncdreamer_")
            tmp_path = os.path.join(tmp_dir.name, "input.png")
            source.convert("RGB").save(tmp_path)

            # Prepare inputs (match training resolution; keep square & divisible by 64)
            data: Dict[str, Any] = prepare_inputs(
                tmp_path,
                self.elevation,
                crop_size=self.res,
                image_size=self.res,
            )

            # Move tensors to device & add batch dimension; repeat for sample_num
            for k, v in list(data.items()):
                if torch.is_tensor(v):
                    v = v.unsqueeze(0).to(self.device, non_blocking=True)
                    v = torch.repeat_interleave(v, sample_num, dim=0)
                    data[k] = v
                else:
                    # leave non-tensors (e.g., metadata) as-is
                    data[k] = v

            # Build sampler (DDIM only per current generator.py)
            if self.default_sampler != "ddim":
                raise NotImplementedError("Only DDIM sampler is supported for now.")
            sampler = SyncDDIMSampler(self.model, self.sample_steps)

            with vram_guard():
                x_sample = self.model.sample(
                    sampler,
                    data,
                    self.cfg_scale,
                    batch_view_num=int(num_views),
                )

            # x_sample: [B, N, C, H, W] in [-1, 1]
            if x_sample.ndim != 5:
                raise RuntimeError(f"Unexpected sample shape: {tuple(x_sample.shape)}")

            B, N, C, H, W = x_sample.shape
            # If the model produced more views than requested, truncate; if fewer, keep all.
            N_ret = min(N, int(num_views))
            x_sample = x_sample[:, :N_ret]

            # Convert to uint8 images (0..255), first sample only
            x_sample = (torch.clamp(x_sample, min=-1.0, max=1.0) + 1) * 0.5
            x_sample = x_sample.permute(0, 1, 3, 4, 2).contiguous()  # [B, N, H, W, C]
            arr = (x_sample.detach().cpu().numpy() * 255).astype(np.uint8)

            out: List[Image.Image] = [
                Image.fromarray(arr[0, i], mode="RGB") for i in range(N_ret)
            ]
            return out

        finally:
            # Cleanup temp artifacts
            try:
                if tmp_dir is not None:
                    tmp_dir.cleanup()
            except Exception:
                pass
