"""
clip_ranker.py

Pick the best image for a given prompt using CLIP similarity.

Supports either:
- open_clip (recommended): pip install open-clip-torch
- clip (OpenAI's original): pip install git+https://github.com/openai/CLIP.git

Usage (module):
    from clip_ranker import ClipRanker

    # Reusable instance – remember to close() when done:
    ranker = ClipRanker(device=torch.device("cuda"), model_name="ViT-L-14")
    best_idx, best_path, scores = ranker.pick_best(
        prompt="a polished brass candlestick",
        images=["a.jpg", "b.png", pil_image_object]
    )
    ranker.close()  # <<< frees GPU

    # Context manager (auto free):
    with ClipRanker(device=torch.device("cuda"), model_name="ViT-L-14") as ranker:
        best_idx, best_path, scores = ranker.pick_best(prompt, images)

    # One-shot helper (load → rank → free):
    best_idx, best_path, scores = ClipRanker.pick_best_once(
        prompt="a polished brass candlestick",
        images=["a.jpg", "b.png"],
        device=torch.device("cuda"),
        model_name="ViT-L-14",
        use_fp16=True,
        image_batch_size=16,
    )

CLI:
    python clip_ranker.py "a polished brass candlestick" a.jpg b.png
"""

from __future__ import annotations

import gc
import io
import os
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union, Optional

import torch
from PIL import Image

# ----------------------------
# Utilities
# ----------------------------


def _is_url(s: str) -> bool:
    try:
        u = urllib.parse.urlparse(s)
        return u.scheme in {"http", "https"}
    except Exception:
        return False


def _load_image(obj: Union[str, Image.Image, bytes]) -> Image.Image:
    """
    Accepts a file path, URL, raw bytes, or PIL.Image and returns a RGB PIL.Image.
    """
    if isinstance(obj, Image.Image):
        img = obj
    elif isinstance(obj, (bytes, bytearray)):
        img = Image.open(io.BytesIO(obj))
    elif isinstance(obj, str):
        if _is_url(obj):
            with urllib.request.urlopen(obj) as r:
                img = Image.open(io.BytesIO(r.read()))
        else:
            img = Image.open(obj)
    else:
        raise TypeError(f"Unsupported image type: {type(obj)}")
    return img.convert("RGB")


# ----------------------------
# Backends
# ----------------------------


@dataclass
class _Backend:
    name: str  # "open_clip" or "clip"
    model: torch.nn.Module
    preprocess: any
    tokenizer: any
    device: torch.device
    dtype: torch.dtype


def _try_open_clip(
    model_name: str, device: torch.device, dtype: torch.dtype
) -> Optional[_Backend]:
    try:
        import open_clip  # type: ignore
    except Exception:
        return None
    # Map a few common aliases to open_clip names
    aliases = {
        "ViT-L-14": ("ViT-L-14", "openai"),
        "ViT-B-32": ("ViT-B-32", "openai"),
        "ViT-B-16": ("ViT-B-16", "openai"),
        "xlm-roberta-base-ViT-B-32": ("xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k"),
    }
    oc_name, oc_ckpt = aliases.get(model_name, (model_name, "openai"))
    model, _, preprocess = open_clip.create_model_and_transforms(
        oc_name, pretrained=oc_ckpt, device=device
    )
    if dtype == torch.float16 and device.type == "cpu":
        # float16 on CPU is not supported; fall back to float32
        dtype = torch.float32
    model = model.to(device=device, dtype=dtype)  # type: ignore[arg-type]
    tokenizer = open_clip.get_tokenizer(oc_name)
    return _Backend("open_clip", model, preprocess, tokenizer, device, dtype)


def _try_clip(
    model_name: str, device: torch.device, dtype: torch.dtype
) -> Optional[_Backend]:
    try:
        import clip  # type: ignore
    except Exception:
        return None
    # Map to clip names
    aliases = {
        "ViT-L-14": "ViT-L/14",
        "ViT-B-32": "ViT-B/32",
        "ViT-B-16": "ViT-B/16",
    }
    clip_name = aliases.get(model_name, model_name)
    model, preprocess = clip.load(clip_name, device=device, jit=False)
    if dtype == torch.float16 and device.type == "cpu":
        dtype = torch.float32
    model = model.to(device=device, dtype=dtype)
    tokenizer = clip.tokenize
    return _Backend("clip", model, preprocess, tokenizer, device, dtype)


# ----------------------------
# Ranker
# ----------------------------


class ClipRanker:
    """
    Rank images by CLIP similarity to a text prompt.
    - Precomputes text embedding once
    - Batches image encoding for speed
    - Returns cosine similarity scores (higher is better)
    - Provides .close() / context manager to free GPU memory
    """

    def __init__(
        self,
        device: torch.device,
        model_name: str = "ViT-L-14",
        use_fp16: bool = True,
        image_batch_size: int = 16,
    ):
        self._closed = False
        self.device = device
        self.dtype = (
            torch.float16
            if (use_fp16 and self.device.type == "cuda")
            else torch.float32
        )
        self.image_batch_size = int(image_batch_size)

        # Prefer open_clip, then fall back to clip
        self.backend = _try_open_clip(model_name, self.device, self.dtype) or _try_clip(
            model_name, self.device, self.dtype
        )
        if self.backend is None:
            raise RuntimeError(
                "No CLIP backend found. Install one of:\n"
                "  - pip install open-clip-torch\n"
                "  - pip install git+https://github.com/openai/CLIP.git"
            )
        self.model_name = model_name
        self.model = self.backend.model.eval()
        self.preprocess = self.backend.preprocess
        self.backend_name = self.backend.name

    # ---------- Lifecycle / Cleanup ----------

    def close(self, aggressively: bool = True) -> None:
        """
        Free GPU/CPU memory used by the model and intermediate tensors.
        If 'aggressively' is True, also calls gc.collect() and CUDA IPC cleanup.
        """
        if self._closed:
            return

        # Drop references first so CUDA allocator sees freeable blocks
        try:
            # Remove model & big attributes
            model = getattr(self, "model", None)
            if model is not None:
                # Some CLIP backends keep buffers on device; move to cpu first to help release
                try:
                    model.to("cpu")
                except Exception:
                    pass
                delattr(self, "model")

            # Preprocess/tokenizer are CPU-side, but drop references regardless
            if hasattr(self, "preprocess"):
                delattr(self, "preprocess")
            if hasattr(self, "backend"):
                delattr(self, "backend")

        except Exception:
            # Ensure we still attempt cache clears
            pass

        # Synchronize & clear CUDA
        if self.device.type == "cuda":
            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                # device may already be torn down; ignore
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if aggressively:
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

        if aggressively:
            # Clear Python-side refs and run GC
            gc.collect()

        self._closed = True

    def __enter__(self) -> "ClipRanker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(aggressively=True)

    def __del__(self):
        # Best-effort cleanup; avoid raising in GC
        try:
            self.close(aggressively=False)
        except Exception:
            pass

    # ---------- Encoding / Scoring ----------

    @torch.inference_mode()
    def _encode_text(self, prompt: str) -> torch.Tensor:
        if self.backend_name == "open_clip":
            tokens = self.backend.tokenizer([prompt])
            tokens = tokens.to(self.device)
            txt = self.model.encode_text(tokens)
        else:
            tokens = self.backend.tokenizer([prompt]).to(self.device)
            txt = self.model.encode_text(tokens)  # type: ignore[attr-defined]
        txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-8)
        return txt.to(self.dtype)

    @torch.inference_mode()
    def _encode_images(
        self, images: Sequence[Union[str, Image.Image, bytes]]
    ) -> torch.Tensor:
        # Preprocess in batches to avoid memory spikes
        embs: List[torch.Tensor] = []
        N = len(images)
        for i in range(0, N, self.image_batch_size):
            batch_imgs = [
                self.preprocess(_load_image(x))
                for x in images[i : i + self.image_batch_size]
            ]
            batch = torch.stack(batch_imgs).to(self.device, dtype=self.dtype)
            img = self.model.encode_image(batch)
            img = img / (img.norm(dim=-1, keepdim=True) + 1e-8)
            embs.append(img)
            # Free per-batch tensors ASAP
            del batch, batch_imgs
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
        return torch.cat(embs, dim=0)

    @torch.inference_mode()
    def score(
        self,
        prompt: str,
        images: Sequence[Union[str, Image.Image, bytes]],
    ) -> List[float]:
        """
        Returns cosine similarity scores (higher is better), one per image.
        """
        if len(images) == 0:
            return []

        t = self._encode_text(prompt)  # [1, D]
        v = self._encode_images(images)  # [N, D]
        sims = (v @ t.T).squeeze(-1)  # [N]
        out = sims.float().tolist()

        # Explicitly drop big tensors ASAP
        del t, v, sims
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

        return out

    @torch.inference_mode()
    def pick_best(
        self,
        prompt: str,
        images: Sequence[Union[str, Image.Image, bytes]],
        return_scores: bool = True,
    ) -> Tuple[int, Optional[str], List[float]]:
        """
        Rank and choose the best image.

        Returns:
            best_idx: index of the best image in the input list (or -1 if empty)
            best_path: if the corresponding input is a str path/URL, returns it; else None
            scores: list of similarity scores (higher is better), same order as inputs
        """
        scores = self.score(prompt, images)
        if not scores:
            return -1, None, []
        best_idx = int(torch.tensor(scores).argmax().item())
        best_raw = images[best_idx]
        best_path = best_raw if isinstance(best_raw, str) else None

        return best_idx, best_path, scores
