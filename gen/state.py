from __future__ import annotations
import base64
import io
from typing import Dict, Tuple, Optional, List
from loguru import logger
import asyncio, time, random, functools
import numpy as np
import torch
from PIL import Image
import os

from gen.settings import Config
from gen.pipelines.t2i_sd35 import SD35Text2Image
from gen.pipelines.bg_birefnet import BiRefNetRemover
from gen.pipelines.i23d_hunyuan import HunYuanImageTo3D
from gen.validators.external_validator import ExternalValidator


class MinerState:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Pipelines pinned to devices
        self.t2i = SD35Text2Image(self.device)
        self.bg_remover = BiRefNetRemover(self.device)
        self.i23d = HunYuanImageTo3D(self.device)

        # External validator
        self.validator = ExternalValidator(
            cfg.validator_url_txt, cfg.validator_url_img, cfg.vld_threshold
        )

        # Knobs
        self.sd35_max_tries: int = getattr(cfg, "sd35_max_tries", 3)
        self.hunyuan_max_tries: int = getattr(cfg, "hunyuan_max_tries", 1)
        self.early_stop_score: float = getattr(
            cfg, "early_stop_score", max(0.0, cfg.vld_threshold)
        )
        self.time_budget_s: Optional[float] = getattr(cfg, "time_budget_s", None)

        # Concurrency
        self.queue_maxsize: int = getattr(cfg, "queue_maxsize", 4)

        # Debug
        self.debug_save: bool = bool(getattr(cfg, "debug_save", False))

        logger.info(f"Models loaded.")

    # ---------------------------
    # Helpers
    # ---------------------------

    def _log_cuda(self, message: str, device: torch.device):
        logger.warning(f"[{device}] {message}")
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def _t2i_param_sweep(self) -> List[Dict]:
        base_steps = self.cfg.sd35_steps
        base_res = self.cfg.sd35_res

        tries: List[Dict] = []
        for i in range(self.sd35_max_tries):
            steps = max(1, base_steps + (i % 2))  # 1 step alternation
            tries.append(
                {
                    "steps": steps,
                    "res": base_res,
                    "seed": random.randint(0, 2**31 - 1),
                }
            )
        return tries

    def _hunyuan_param_sweep(self) -> List[Dict]:
        params = []
        for _ in range(max(1, self.hunyuan_max_tries)):
            params.append(
                {
                    "seed": random.randint(0, 2**31 - 1),
                }
            )
        return params

    def _gen_one_image(self, prompt: str, params: dict):
        try:
            img = self.t2i.generate(
                prompt=prompt,
                steps=params["steps"],
                res=params["res"],
                seed=params["seed"],
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("SD3.5 OOM; clearing cache and skipping one try.")
                if self.device.type == "cuda":
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                return None, params
            raise
        return img, params

    async def _bg_remove_one(self, pil_image: Image.Image) -> Image.Image | None:
        try:
            fg, _ = self.bg_remover.remove(pil_image)
            return fg
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("BiRefNet OOM; clearing cache and dropping item.")
                if self.device.type == "cuda":
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                return None
            raise

    async def _hunyuan_one(self, pil_image, params: dict):
        try:
            ply_bytes = self.i23d.infer_to_ply(
                pil_image,
                seed=params.get("seed"),
            )
            return ply_bytes, params
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self._log_cuda("HunYuan OOM; drop this try.", self.device)
                return b"", params
            # Unknown runtime — return empty to keep the pipeline flowing
            self._log_cuda(f"HunYuan runtime error: {e}", self.device)
            return b"", params

    def _within_budget(self, start_ts: float) -> bool:
        if self.time_budget_s is None:
            return True
        return (time.time() - start_ts) < self.time_budget_s

    # -------------------------------------------------
    # Process 1: SD35 | ==> q01
    # -------------------------------------------------
    async def _proc1_t2i(self, prompt, q01, start_ts, stop_evt):
        tasks: List[asyncio.Task] = []

        async def _one(params: dict):
            if stop_evt.is_set() or not self._within_budget(start_ts):
                return
            t0 = time.time()
            img, iparams = self._gen_one_image(prompt, params)
            t2i_sec = time.time() - t0
            if img is None:
                return

            logger.debug(f"[Proc1] T2I {t2i_sec:.2f}s -> q01")
            await q01.put((img, iparams))

        try:
            for p in self._t2i_param_sweep():
                if stop_evt.is_set() or not self._within_budget(start_ts):
                    break
                tasks.append(asyncio.create_task(_one(p)))

            for t in asyncio.as_completed(tasks):
                await t
        finally:
            await q01.put(None)  # sentinel to Process 2

    # -------------------------------------------------
    # Process 2: BiRefNet + HunYuan | q01 ==> q12
    # -------------------------------------------------
    async def _proc2_bg_i23d(self, q01, q12, start_ts, stop_evt):
        # single-worker default; bump if VRAM allows (careful!)
        while True:
            if stop_evt.is_set() or not self._within_budget(start_ts):
                break
            item = await q01.get()
            if item is None:
                # propagate sentinel downstream
                await q12.put(None)
                break

            img, iparams = item

            t1 = time.time()
            fg = await self._bg_remove_one(img)
            bg_sec = time.time() - t1
            if fg is None:
                return

            if self.debug_save:
                fg.save(os.path.join("out", f"input_{iparams['seed']}.png"))

            if stop_evt.is_set() or not self._within_budget(start_ts):
                break
            hparams = self._hunyuan_param_sweep()[0]
            t0 = time.time()
            ply_bytes, _ = await self._hunyuan_one(fg, hparams)
            i23d_sec = time.time() - t0

            if not ply_bytes:
                logger.debug("[Proc2] Empty PLY; dropping.")
                continue

            logger.debug(f"[Proc2] BG {bg_sec:.sf}s + HunYuan {i23d_sec:.2f}s -> q12")
            # Attach metadata for logging/inspection
            await q12.put((ply_bytes, {"iparams": iparams, "hparams": hparams}))

    # -------------------------------------------------
    # Process 3: Validate | q12
    # -------------------------------------------------
    async def _proc3_validate(
        self,
        prompt: str,
        q12: asyncio.Queue,
        start_ts: float,
        stop_evt: asyncio.Event,
        best_out: Dict[str, object],
    ):
        best_score = -1.0
        best_ply: bytes = b""

        while True:
            if stop_evt.is_set() or not self._within_budget(start_ts):
                break

            item = await q12.get()
            if item is None:
                break  # Process 2 finished and propagated sentinel

            ply_bytes, meta = item
            iparams = meta.get("iparams", {})
            hparams = meta.get("hparams", {})

            v0 = time.time()
            # If your validator uses GPU0, this now runs with GPU0 free of Flux/BG.
            score, passed, _ = await self.validator.validate_text(prompt, ply_bytes)

            vsec = time.time() - v0
            logger.info(
                f"[Proc3] T2I{iparams}|I23D{hparams} -> score={score:.4f}, "
                f"passed={passed} (validate {vsec:.2f}s)"
            )

            if score > best_score:
                best_score, best_ply = score, ply_bytes

            if score >= self.early_stop_score:
                logger.info(
                    f"[Proc3] Early-stop at score {score:.4f} >= {self.early_stop_score:.4f}"
                )
                best_out["ply"] = ply_bytes if passed else b""
                best_out["score"] = score
                stop_evt.set()
                # We won’t cancel HunYuan actively; queue will drain to sentinel.
                break

        final_pass = best_score >= self.cfg.vld_threshold
        best_out.setdefault("ply", best_ply if final_pass else b"")
        best_out.setdefault("score", max(0.0, best_score))

    # ---------------------------
    # Public APIs
    # ---------------------------

    async def text_to_ply(self, prompt: str) -> Tuple[bytes, float]:
        """
        3-stage pipeline with two queues:
          Proc1: T2I -> q01
          Proc2: BG + HunYuan consumes q01 -> q12
          Proc3: Validates items from q12
        """
        start_ts = time.time()
        q01: asyncio.Queue = asyncio.Queue(maxsize=self.queue_maxsize)
        q12: asyncio.Queue = asyncio.Queue(maxsize=self.queue_maxsize)
        stop_evt = asyncio.Event()
        best_out: Dict[str, object] = {}

        proc1 = asyncio.create_task(self._proc1_t2i(prompt, q01, start_ts, stop_evt))
        proc2 = asyncio.create_task(self._proc2_bg_i23d(q01, q12, start_ts, stop_evt))
        proc3 = asyncio.create_task(
            self._proc3_validate(prompt, q12, start_ts, stop_evt, best_out)
        )

        try:
            await asyncio.gather(proc1, proc2, proc3)
        finally:
            # Drain any leftover items
            for q in (q01, q12):
                try:
                    while not q.empty():
                        _ = q.get_nowait()
                except Exception:
                    pass

        ply_bytes: bytes = best_out.get("ply", b"")  # type: ignore
        score: float = float(best_out.get("score", 0.0))  # type: ignore
        return ply_bytes, score

    async def image_to_ply(self, image_b64) -> Tuple[bytes, float]:
        """
        Image mode remains sequential per image.
        """
        start_ts = time.time()

        try:
            raw = base64.b64decode(image_b64, validate=False)
        except Exception:
            raw = base64.b64decode(image_b64 or "")
        pil_image = Image.open(io.BytesIO(raw)).convert("RGBA")

        # BG removal
        fg, _ = self.bg_remover.remove(pil_image)

        best_score = -1.0
        best_ply: bytes = b""

        for hparams in self._hunyuan_param_sweep():
            if not self._within_budget(start_ts):
                logger.warning("Budget exhausted mid-HunYuan(image); stopping.")
                break

            ply_bytes, _ = await self._hunyuan_one(fg, hparams)
            if not ply_bytes:
                continue

            score, passed, _ = await self.validator.validate_image(image_b64, ply_bytes)
            logger.info(
                f"[image] HunYuan{hparams} -> score={score:.4f}, passed={passed}"
            )

            if score > best_score:
                best_score, best_ply = score, ply_bytes

            if score >= self.early_stop_score:
                logger.info(
                    f"[image] Early-stop: score {score:.4f} >= {self.early_stop_score:.4f}"
                )
                return (ply_bytes if passed else b""), score

        final_pass = best_score >= self.cfg.vld_threshold
        return (best_ply if final_pass else b""), max(0.0, best_score)
