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

    def _generate_multiviews(self, rgb_img: Image.Image) -> List[Image.Image]:
        return self.mv.generate_views(
            source=rgb_img,
            num_views=self.mv_num_views,
            seed=random.randint(0, 2**31 - 1),
        )

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
                if self.debug_save:
                    self._save_pil(base_img, "t2i_base")

            # 2) BG removal FIRST (on base)
            if not self._within_budget(start_ts):
                try:
                    base_img.close()
                except Exception:
                    pass
                del base_img
                logger.warning("Budget exhausted pre-BG; stopping.")
                break

            with vram_guard():
                # t0 = _time.time()
                # base_fg = await self._bg_remove_one(base_img)  # RGBA (foreground)
                # logger.debug(f"BG remove (base): {_time.time() - t0:.2f}s")
                # if self.debug_save:
                #     self._save_pil(base_fg, "t2i_base_fg")
                base_fg = base_img.copy()

            # 3) base_fg -> multiview (drives model with already-cut subject)
            if not self._within_budget(start_ts):
                try:
                    base_img.close()
                except Exception:
                    pass
                del base_img
                try:
                    base_fg.close()
                except Exception:
                    pass
                del base_fg
                logger.warning("Budget exhausted pre-MV (after BG); stopping.")
                break

            with vram_guard():
                t0 = _time.time()
                mv_imgs = [base_fg.copy()]
                mv_imgs += self._generate_multiviews(base_fg)
                logger.debug(f"MV: {_time.time() - t0:.2f}s")
                if self.debug_save:
                    for i, im in enumerate(mv_imgs):
                        self._save_pil(im, f"mv_after_bg_{i:02d}")

            try:
                base_img.close()
            except Exception:
                pass
            del base_img
            try:
                base_fg.close()
            except Exception:
                pass
            del base_fg

            # 4) Rank & select top-K directly on MV outputs
            if not self._within_budget(start_ts):
                logger.warning("Budget exhausted pre-ranking; stopping.")
                for im in mv_imgs:
                    try:
                        im.close()
                    except Exception:
                        pass
                del mv_imgs
                break

            ranked = score_views(mv_imgs)
            if self.cfg.debug_save:
                logger.debug(f"View scores (top 8): {ranked[:8]}")
                # full ranking dump
                self._save_json({"ranked": ranked}, "mv_ranked")
            top_indices = [i for (i, _) in ranked[: self.mv_top_k_for_trellis]]

            # Ensure RGBA input for Trellis if it expects 4-channel
            top_views = []
            for i in top_indices:
                im = mv_imgs[i]
                if im.mode != "RGBA":
                    im = im.convert("RGBA")
                top_views.append(im)

            if self.debug_save:
                for rpos, i in enumerate(top_indices):
                    self._save_pil(mv_imgs[i], f"mv_top_{rpos:02d}_idx_{i:02d}")

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
                if self.debug_save:
                    # keep each candidate with score/flag
                    tag = f"textply_score_{score:.4f}_{'pass' if passed else 'fail'}"
                    self._save_bytes(ply_bytes, tag, ".ply")

                if score > best_score:
                    best_score, best_ply = score, ply_bytes

                if score >= self.early_stop_score:
                    logger.info(
                        f"[text] Early-stop: score {score:.4f} >= {self.early_stop_score:.4f}"
                    )
                    # free
                    for im in mv_imgs:
                        try:
                            im.close()
                        except Exception:
                            pass
                    del mv_imgs
                    return (ply_bytes if passed else b""), score

            # free between outer tries
            for im in mv_imgs:
                try:
                    im.close()
                except Exception:
                    pass
            del mv_imgs

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
        if self.debug_save:
            self._save_pil(base_img, "input_image_rgb")

        # 1) BG removal FIRST on the input image
        with vram_guard():
            # t0 = _time.time()
            # base_fg = await self._bg_remove_one(base_img)  # RGBA foreground
            # logger.debug(f"BG remove (input): {_time.time() - t0:.2f}s")
            # if self.debug_save:
            #     self._save_pil(base_fg, "input_image_fg")
            base_fg = base_img.copy()

        # 2) base_fg -> multiview (include base_fg as candidate 0)
        with vram_guard():
            t0 = _time.time()
            mv_imgs = [base_fg.copy()]
            mv_imgs += self._generate_multiviews(base_fg, base_prompt=None)
            logger.debug(f"MV: {_time.time() - t0:.2f}s")
            if self.debug_save:
                for i, im in enumerate(mv_imgs):
                    self._save_pil(im, f"mv_after_bg_{i:02d}")

        try:
            base_img.close()
        except Exception:
            pass
        del base_img
        try:
            base_fg.close()
        except Exception:
            pass
        del base_fg

        # 3) Rank & select top-K (no per-view BG now)
        if not self._within_budget(start_ts):
            logger.warning("Budget exhausted pre-ranking; stopping.")
            for im in mv_imgs:
                try:
                    im.close()
                except Exception:
                    pass
            del mv_imgs
            return b"", 0.0

        ranked = score_views(mv_imgs)
        if self.cfg.debug_save:
            logger.debug(f"View scores (top 8): {ranked[:8]}")
            self._save_json({"ranked": ranked}, "mv_ranked_image")
        top_indices = [i for (i, _) in ranked[: self.mv_top_k_for_trellis]]

        # Ensure Trellis input is RGBA
        top_views: List[Image.Image] = []
        for i in top_indices:
            im = mv_imgs[i]
            if im.mode != "RGBA":
                im = im.convert("RGBA")
            top_views.append(im)

        if self.debug_save:
            for rpos, i in enumerate(top_indices):
                self._save_pil(mv_imgs[i], f"mv_top_{rpos:02d}_idx_{i:02d}")

        # 4) Trellis param sweep + validation against the original image
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
            # Validate with the input image
            score, passed, _ = await self.validator.validate_image(image_b64, ply_bytes)
            logger.info(
                f"Validate (image): score={score:.4f}, passed={passed}, {_time.time() - t0:.2f}s"
            )

            if self.debug_save:
                tag = f"imgply_score_{score:.4f}_{'pass' if passed else 'fail'}"
                self._save_bytes(ply_bytes, tag, ".ply")

            if score > best_score:
                best_score, best_ply = score, ply_bytes

            if score >= self.early_stop_score:
                logger.info(
                    f"[image] Early-stop: score {score:.4f} >= {self.early_stop_score:.4f}"
                )
                for im in mv_imgs:
                    try:
                        im.close()
                    except Exception:
                        pass
                del mv_imgs
                return (ply_bytes if passed else b""), score

        # Cleanup
        for im in mv_imgs:
            try:
                im.close()
            except Exception:
                pass
        del mv_imgs

        final_pass = best_score >= self.cfg.vld_threshold
        return (best_ply if final_pass else b""), max(0.0, best_score)
