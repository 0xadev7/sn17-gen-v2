from __future__ import annotations
import base64
import io
import os
from typing import Dict, Tuple, Optional, List
from loguru import logger
import time, random
import torch
from PIL import Image
from pathlib import Path
import json

from gen.settings import Config

from gen.pipelines.bg_birefnet import BiRefNetRemover
from gen.pipelines.i23d_trellis import TrellisImageTo3D
from gen.utils.clip_ranker import ClipRanker
from gen.validators.external_validator import ExternalValidator
from gen.utils.vram import vram_guard


class MinerState:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Pipelines pinned to devices
        if self.cfg.t2i_backend == "sd35":
            from gen.pipelines.t2i_sd35 import SD35Text2Image

            self.t2i = SD35Text2Image(self.device)
        elif self.cfg.t2i_backend == "flux":
            from gen.pipelines.t2i_flux import FluxText2Image

            self.t2i = FluxText2Image(self.device)
        else:
            logger.error(f"Unknown T2I backend: {self.cfg.t2i_backend}")

        self.ranker = ClipRanker(self.device)
        self.bg_remover = BiRefNetRemover(self.device)

        if self.cfg.mv_backend == "sync_dreamer":
            from gen.pipelines.i2mv_sync_dreamer import SyncDreamerMV

            self.mv = SyncDreamerMV(
                self.device,
                res=self.cfg.mv_res,
                elevation=cfg.sync_dreamer_elevation,
                sample_steps=cfg.sync_dreamer_sample_steps,
                cfg_scale=cfg.sync_dreamer_cfg_scale,
            )
        elif self.cfg.mv_backend == "zero123":
            from gen.pipelines.i2mv_zero123 import Zero123MV

            self.mv = Zero123MV(self.device, self.cfg.mv_res)
        else:
            logger.error(f"Unknown MV backend: {self.cfg.mv_backend}")

        self.trellis_img = TrellisImageTo3D(self.device)

        # External validator
        self.validator = ExternalValidator(
            cfg.validator_url_txt, cfg.validator_url_img, cfg.vld_threshold
        )

        # Debug
        self.debug_save: bool = bool(getattr(cfg, "debug_save", False))
        # temp dump dir
        self._tmp_dir: Optional[Path] = None
        self._save_idx: int = 0
        if self.debug_save:
            self._tmp_dir = Path("./tmp")
            try:
                self._tmp_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Failed to create ./tmp: {e}")

        # Multi-view settings
        self.mv_num_views: int = max(1, int(getattr(cfg, "mv_num_views", 4)))
        yaws_csv = (getattr(cfg, "mv_yaws_csv", "") or "").strip()
        if yaws_csv:
            try:
                self.mv_yaws = [float(v) for v in yaws_csv.split(",") if v.strip()]
            except Exception:
                self.mv_yaws = []
        else:
            self.mv_yaws = []

        logger.info("Models loaded (T2I / Ranker / BG / MV / Trellis).")

    # ---------------------------
    # Helpers
    # ---------------------------

    def _next_name(self, kind: str, ext: str) -> Optional[Path]:
        if not self.debug_save or not self._tmp_dir:
            return None
        ts = int(time.time())
        self._save_idx += 1
        return self._tmp_dir / f"{ts:010d}_{self._save_idx:04d}_{kind}{ext}"

    def _save_pil(self, img: Image.Image, kind: str) -> None:
        p = self._next_name(kind, ".png")
        if p is None:
            return
        try:
            img.save(p)
            logger.debug(f"[debug_save] saved image: {p}")
        except Exception as e:
            logger.debug(f"[debug_save] failed to save image {p}: {e}")

    def _save_bytes(self, data: bytes, kind: str, ext: str) -> Optional[Path]:
        p = self._next_name(kind, ext)
        if p is None:
            return None
        try:
            with open(p, "wb") as f:
                f.write(data)
            logger.debug(f"[debug_save] saved bytes: {p}")
        except Exception as e:
            logger.debug(f"[debug_save] failed to save bytes {p}: {e}")
        return p

    def _save_json(self, obj: dict, kind: str) -> None:
        p = self._next_name(kind, ".json")
        if p is None:
            return
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            logger.debug(f"[debug_save] saved json: {p}")
        except Exception as e:
            logger.debug(f"[debug_save] failed to save json {p}: {e}")

    def _t2i_param_sweep(self) -> List[Dict]:
        base_steps = self.cfg.t2i_steps
        base_res = self.cfg.t2i_res

        tries: List[Dict] = []
        for i in range(3):
            steps = max(1, base_steps + (i % 2))
            tries.append(
                {
                    "steps": steps,
                    "res": base_res,
                    "seed": random.randint(0, 2**31 - 1),
                }
            )
        return tries

    # ---------------------------
    # Public APIs
    # ---------------------------

    async def text_to_ply(self, prompt: str) -> Tuple[bytes, float]:
        import time as _time

        # 1) Generate base images
        base_images = []
        for params in self._t2i_param_sweep():
            with vram_guard():
                t0 = _time.time()
                base_img = self.t2i.generate(
                    prompt,
                    steps=params["steps"],
                    res=params["res"],
                    seed=params["seed"],
                )
                logger.debug(f"T2I: {_time.time() - t0:.2f}s")
                base_images.append(base_img)
                if self.debug_save:
                    self._save_pil(base_img, f"t2i_base_{params['seed']}")

        # 2) Pick the best image
        if len(base_images) == 0:
            logger.warning("No base images generated; aborting.")
            return b"", 0.0

        with vram_guard():
            t0 = _time.time()
            best_idx, _, _ = self.ranker.pick_best(prompt=prompt, images=base_images)
            logger.debug(f"CLIP rank (base): {_time.time() - t0:.2f}s")

            if best_idx < 0:
                logger.warning("Ranking failed; aborting.")
                for im in base_images:
                    try:
                        im.close()
                    except Exception:
                        pass
                del base_images
                return b"", 0.0

        base_img = base_images[best_idx].copy()
        if self.debug_save:
            self._save_pil(base_img, "t2i_best_base")

        for im in base_images:
            try:
                im.close()
            except Exception:
                pass
        del base_images

        # 3) BG removal
        with vram_guard():
            t0 = _time.time()
            base_fg, _ = self.bg_remover.remove(base_img)
            logger.debug(f"BG remove (base): {_time.time() - t0:.2f}s")
            if self.debug_save:
                self._save_pil(base_fg, "t2i_base_fg")
            base_fg = base_img.copy()

        try:
            base_img.close()
        except Exception:
            pass
        del base_img

        # 4) Multi-view generation
        with vram_guard():
            t0 = _time.time()
            mv_imgs = self.mv.generate_views(
                source=base_fg,
                num_views=self.mv_num_views,
                seed=random.randint(0, 2**31 - 1),
            )
            logger.debug(f"MV: {_time.time() - t0:.2f}s")
            if self.debug_save:
                for i, im in enumerate(mv_imgs):
                    self._save_pil(im, f"mv_after_bg_{i:02d}")

        try:
            base_fg.close()
        except Exception:
            pass
        del base_fg

        # 5) Trellis
        with vram_guard(ipc_collect=True):
            t0 = _time.time()
            ply_bytes = self.trellis_img.infer_multiview_to_ply(
                images=mv_imgs,
                struct_steps=self.cfg.trellis_struct_steps,
                slat_steps=self.cfg.trellis_slat_steps,
                cfg_struct=self.cfg.trellis_cfg_struct,
                cfg_slat=self.cfg.trellis_cfg_slat,
                seed=random.randint(0, 2**31 - 1),
            )
            logger.debug(f"Trellis: {_time.time() - t0:.2f}s")

        for im in mv_imgs:
            try:
                im.close()
            except Exception:
                pass
        del mv_imgs

        # 6) Validation
        if not ply_bytes:
            logger.warning("No PLY generated; aborting.")
            return b"", 0.0

        t0 = _time.time()
        score, passed, _ = await self.validator.validate_text(prompt, ply_bytes)
        logger.info(
            f"Validate: score={score:.4f}, passed={passed}, {_time.time() - t0:.2f}s (MV batch)"
        )
        if self.debug_save:
            tag = f"textply_score_{score:.4f}_{'pass' if passed else 'fail'}"
            self._save_bytes(ply_bytes, tag, ".ply")

        final_pass = score >= self.cfg.vld_threshold
        return (ply_bytes if final_pass else b""), max(0.0, score)

    async def image_to_ply(self, image_b64) -> Tuple[bytes, float]:
        import time as _time

        # 1) decode → RGB PIL
        try:
            raw = base64.b64decode(image_b64, validate=False)
        except Exception:
            raw = base64.b64decode(image_b64 or "")
        base_img = Image.open(io.BytesIO(raw)).convert("RGB")
        if self.debug_save:
            self._save_pil(base_img, "input_image_rgb")

        # 2) BG removal
        with vram_guard():
            t0 = _time.time()
            base_fg, _ = self.bg_remover.remove(base_img)
            logger.debug(f"BG remove (input): {_time.time() - t0:.2f}s")
            if self.debug_save:
                self._save_pil(base_fg, "input_image_fg")

        try:
            base_img.close()
        except Exception:
            pass
        del base_img

        # 3) Multi-view generation
        with vram_guard():
            t0 = _time.time()
            mv_imgs = self.mv.generate_views(
                source=base_fg,
                num_views=self.mv_num_views,
                seed=random.randint(0, 2**31 - 1),
            )
            logger.debug(f"MV: {_time.time() - t0:.2f}s")
            if self.debug_save:
                for i, im in enumerate(mv_imgs):
                    self._save_pil(im, f"mv_after_bg_{i:02d}")

        try:
            base_fg.close()
        except Exception:
            pass
        del base_fg

        # 4) Trellis
        with vram_guard(ipc_collect=True):
            t0 = _time.time()
            ply_bytes = self.trellis_img.infer_multiview_to_ply(
                images=mv_imgs,
                struct_steps=self.cfg.trellis_struct_steps,
                slat_steps=self.cfg.trellis_slat_steps,
                cfg_struct=self.cfg.trellis_cfg_struct,
                cfg_slat=self.cfg.trellis_cfg_slat,
                seed=random.randint(0, 2**31 - 1),
            )
            logger.debug(f"Trellis: {_time.time() - t0:.2f}s")

        for im in mv_imgs:
            try:
                im.close()
            except Exception:
                pass
        del mv_imgs

        # 5) Validation
        if not ply_bytes:
            logger.warning("No PLY generated; aborting.")
            return b"", 0.0

        t0 = _time.time()
        score, passed, _ = await self.validator.validate_image(image_b64, ply_bytes)
        logger.info(
            f"Validate: score={score:.4f}, passed={passed}, {_time.time() - t0:.2f}s (MV batch)"
        )
        if self.debug_save:
            tag = f"imgply_score_{score:.4f}_{'pass' if passed else 'fail'}"
            self._save_bytes(ply_bytes, tag, ".ply")

        final_pass = score >= self.cfg.vld_threshold
        return (ply_bytes if final_pass else b""), max(0.0, score)
