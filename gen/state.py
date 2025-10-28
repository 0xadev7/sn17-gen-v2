from __future__ import annotations
import base64
import io
from typing import Dict, Tuple, Optional, List
from loguru import logger
import time, random
import torch
from PIL import Image

from gen.settings import Config

from gen.pipelines.bg_birefnet import BiRefNetRemover
from gen.pipelines.i23d_trellis import TrellisImageTo3D
from gen.validators.external_validator import ExternalValidator
from gen.utils.vram import vram_guard
from gen.utils.select import score_views


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

        self.bg_remover = BiRefNetRemover(self.device)

        if self.cfg.mv_backend == "sync_dreamer":
            from gen.pipelines.i2mv_sync_dreamer import SyncDreamerMV

            self.mv = SyncDreamerMV(self.device, res=self.cfg.mv_res)
        elif self.cfg.mv_backend == "zero123":
            from gen.pipelines.i2mv_zero123 import Zero123MV

            self.mv = Zero123MV(self.device, self.cfg.mv_res)
        elif self.cfg.mv_backend == "sd35":
            from gen.pipelines.i2mv_sd35 import SD35MV

            self.mv = SD35MV(self.device, self.cfg.mv_res)
        else:
            logger.error(f"Unknown MV backend: {self.cfg.mv_backend}")

        self.trellis_img = TrellisImageTo3D(self.device)

        # External validator
        self.validator = ExternalValidator(
            cfg.validator_url_txt, cfg.validator_url_img, cfg.vld_threshold
        )

        # Knobs
        self.t2i_max_tries: int = getattr(cfg, "t2i_max_tries", 3)
        self.trellis_max_tries: int = getattr(cfg, "trellis_max_tries", 1)
        self.early_stop_score: float = getattr(
            cfg, "early_stop_score", max(0.0, cfg.vld_threshold)
        )
        self.time_budget_s: Optional[float] = getattr(cfg, "time_budget_s", None)

        self.debug_save: bool = bool(getattr(cfg, "debug_save", False))

        # Multi-view settings
        self.mv_num_views: int = max(1, int(getattr(cfg, "mv_num_views", 8)))
        self.mv_top_k_for_trellis: int = max(
            1, int(getattr(cfg, "mv_top_k_for_trellis", 4))
        )
        yaws_csv = (getattr(cfg, "mv_yaws_csv", "") or "").strip()
        if yaws_csv:
            try:
                self.mv_yaws = [float(v) for v in yaws_csv.split(",") if v.strip()]
            except Exception:
                self.mv_yaws = []
        else:
            self.mv_yaws = []

        logger.info("Models loaded (T2I / MV / BG / Trellis).")

    # ---------------------------
    # Helpers
    # ---------------------------

    def _t2i_param_sweep(self) -> List[Dict]:
        base_steps = self.cfg.t2i_steps
        base_guidance = self.cfg.t2i_guidance
        base_res = self.cfg.t2i_res

        tries: List[Dict] = []
        for i in range(self.t2i_max_tries):
            steps = max(1, base_steps + (i % 2))
            tries.append(
                {
                    "steps": steps,
                    "guidance": base_guidance,
                    "res": base_res,
                    "seed": random.randint(0, 2**31 - 1),
                }
            )
        return tries

    def _trellis_param_sweep(self) -> List[Dict]:
        params = []
        for _ in range(max(1, self.trellis_max_tries)):
            params.append(
                {
                    "struct_steps": self.cfg.trellis_struct_steps,
                    "slat_steps": self.cfg.trellis_slat_steps,
                    "cfg_struct": self.cfg.trellis_cfg_struct,
                    "cfg_slat": self.cfg.trellis_cfg_slat,
                    "seed": random.randint(0, 2**31 - 1),
                }
            )
        return params

    async def _gen_one_image(self, prompt: str, params: dict):
        return self.t2i.generate(
            prompt,
            steps=params["steps"],
            guidance=params["guidance"],
            res=params["res"],
            seed=params["seed"],
        )

    async def _bg_remove_one(self, pil_image: Image.Image):
        fg, _ = self.bg_remover.remove(pil_image)
        return fg

    def _within_budget(self, start_ts: float) -> bool:
        if self.time_budget_s is None:
            return True
        return (time.time() - start_ts) < self.time_budget_s

    # ---------------------------
    # Multi-view helpers
    # ---------------------------

    def _generate_multiviews(
        self, rgb_img: Image.Image, base_prompt: Optional[str]
    ) -> List[Image.Image]:
        return self.mv.generate_views(
            source=rgb_img,
            num_views=self.mv_num_views,
            seed=random.randint(0, 2**31 - 1),
        )

    def _prep_views_for_trellis(
        self, views_rgb: List[Image.Image]
    ) -> List[Image.Image]:
        rgba_views: List[Image.Image] = []
        for v in views_rgb:
            with vram_guard():
                fg, _ = self.bg_remover.remove(v)
                rgba_views.append(fg)
            try:
                v.close()
            except Exception:
                pass
        return rgba_views

    # ---------------------------
    # Public APIs
    # ---------------------------

    async def text_to_ply(self, prompt: str) -> Tuple[bytes, float]:
        import time as _time

        start_ts = _time.time()
        best_score = -1.0
        best_ply: bytes = b""

        # 1) text -> base image(s)
        for iparams in self._t2i_param_sweep():
            if not self._within_budget(start_ts):
                logger.warning("Budget exhausted mid-T2I; stopping.")
                break

            with vram_guard():
                t0 = _time.time()
                base_img = await self._gen_one_image(prompt, iparams)  # RGB
                logger.debug(f"T2I: {_time.time() - t0:.2f}s")

            # 2) base -> multiview RGB
            if not self._within_budget(start_ts):
                try:
                    base_img.close()
                except Exception:
                    pass
                del base_img
                logger.warning("Budget exhausted pre-MV; stopping.")
                break

            with vram_guard():
                t0 = _time.time()
                mv_rgb = self._generate_multiviews(base_img, base_prompt=prompt)
                logger.debug(f"MV: {_time.time() - t0:.2f}s")

            try:
                base_img.close()
            except Exception:
                pass
            del base_img

            # 3) BG removal per view
            if not self._within_budget(start_ts):
                logger.warning("Budget exhausted pre-BG; stopping.")
                for im in mv_rgb:
                    try:
                        im.close()
                    except Exception:
                        pass
                del mv_rgb
                break
            mv_rgba = self._prep_views_for_trellis(mv_rgb)

            # 4) Rank & select top-K (to feed Trellis at once)
            ranked = score_views(mv_rgba)
            if self.cfg.debug_save:
                logger.debug(f"View scores (top 8): {ranked[:8]}")
            top_indices = [i for (i, _) in ranked[: self.mv_top_k_for_trellis]]
            top_views = [mv_rgba[i] for i in top_indices]

            # 5) Trellis (single call per param set) + validation
            for tparams in self._trellis_param_sweep():
                if not self._within_budget(start_ts):
                    logger.warning("Budget exhausted mid-Trellis; stopping.")
                    break

                with vram_guard(ipc_collect=True):
                    t0 = _time.time()
                    ply_bytes = self.trellis_img.infer_multiview_to_ply(
                        images=top_views,
                        struct_steps=tparams["struct_steps"],
                        slat_steps=tparams["slat_steps"],
                        cfg_struct=tparams["cfg_struct"],
                        cfg_slat=tparams["cfg_slat"],
                        seed=tparams.get("seed"),
                    )
                    logger.debug(
                        f"Trellis (MV x{len(top_views)}): {_time.time() - t0:.2f}s"
                    )

                if not ply_bytes:
                    continue

                if not self._within_budget(start_ts):
                    logger.warning("Budget exhausted mid-Validation; stopping.")
                    break

                t0 = _time.time()
                score, passed, _ = await self.validator.validate_text(prompt, ply_bytes)
                logger.info(
                    f"Validate: score={score:.4f}, passed={passed}, {_time.time() - t0:.2f}s (MV batch)"
                )

                if score > best_score:
                    best_score, best_ply = score, ply_bytes

                if score >= self.early_stop_score:
                    logger.info(
                        f"[text] Early-stop: score {score:.4f} >= {self.early_stop_score:.4f}"
                    )
                    # free
                    for im in mv_rgba:
                        try:
                            im.close()
                        except Exception:
                            pass
                    del mv_rgba
                    return (ply_bytes if passed else b""), score

            # free between outer tries
            for im in mv_rgba:
                try:
                    im.close()
                except Exception:
                    pass
            del mv_rgba

        final_pass = best_score >= self.cfg.vld_threshold
        return (best_ply if final_pass else b""), max(0.0, best_score)

    async def image_to_ply(self, image_b64) -> Tuple[bytes, float]:
        import time as _time

        start_ts = _time.time()
        best_score = -1.0
        best_ply: bytes = b""

        # decode → RGB PIL
        try:
            raw = base64.b64decode(image_b64, validate=False)
        except Exception:
            raw = base64.b64decode(image_b64 or "")
        base_img = Image.open(io.BytesIO(raw)).convert("RGB")

        # 1) base -> multiview (include base as candidate 0)
        with vram_guard():
            t0 = _time.time()
            mv_rgb = [base_img.copy()]
            mv_rgb += self._generate_multiviews(base_img, base_prompt=None)
            logger.debug(f"MV: {_time.time() - t0:.2f}s")

        try:
            base_img.close()
        except Exception:
            pass
        del base_img

        # 2) BG removal
        if not self._within_budget(start_ts):
            logger.warning("Budget exhausted pre-BG; stopping.")
            for im in mv_rgb:
                try:
                    im.close()
                except Exception:
                    pass
            del mv_rgb
            return b"", 0.0

        mv_rgba = self._prep_views_for_trellis(mv_rgb)

        # 3) Rank & select top-K
        ranked = score_views(mv_rgba)
        top_indices = [i for (i, _) in ranked[: self.mv_top_k_for_trellis]]
        top_views = [mv_rgba[i] for i in top_indices]

        # 4) Trellis (single call per param set) + validation
        for tparams in self._trellis_param_sweep():
            if not self._within_budget(start_ts):
                logger.warning("Budget exhausted mid-Trellis; stopping.")
                break

            with vram_guard(ipc_collect=True):
                t0 = _time.time()
                ply_bytes = self.trellis_img.infer_multiview_to_ply(
                    images=top_views,
                    struct_steps=tparams["struct_steps"],
                    slat_steps=tparams["slat_steps"],
                    cfg_struct=tparams["cfg_struct"],
                    cfg_slat=tparams["cfg_slat"],
                    seed=tparams.get("seed"),
                )
                logger.debug(
                    f"Trellis (MV x{len(top_views)}): {_time.time() - t0:.2f}s"
                )

            if not ply_bytes:
                continue

            t0 = _time.time()
            score, passed, _ = await self.validator.validate_image(image_b64, ply_bytes)
            logger.info(
                f"Validate: score={score:.4f}, passed={passed}, {_time.time() - t0:.2f}s (MV batch)"
            )

            if score > best_score:
                best_score, best_ply = score, ply_bytes

            if score >= self.early_stop_score:
                logger.info(
                    f"[image] Early-stop: score {score:.4f} >= {self.early_stop_score:.4f}"
                )
                for im in mv_rgba:
                    try:
                        im.close()
                    except Exception:
                        pass
                del mv_rgba
                return (ply_bytes if passed else b""), score

        # free
        for im in mv_rgba:
            try:
                im.close()
            except Exception:
                pass
        del mv_rgba

        final_pass = best_score >= self.cfg.vld_threshold
        return (best_ply if final_pass else b""), max(0.0, best_score)

    def close(self):
        # drop big refs
        for obj in [self.t2i, self.bg_remover, self.mv, self.trellis_img]:
            try:
                del obj
            except Exception:
                pass
        import gc, torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
