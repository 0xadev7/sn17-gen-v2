from __future__ import annotations
import base64
import io
from typing import Dict, Tuple, Optional, List
from loguru import logger
import asyncio, time, random, functools
import numpy as np
import torch
from PIL import Image

from gen.settings import Config
from gen.pipelines.t2i_sd35 import SD35Text2Image
from gen.pipelines.bg_birefnet import BiRefNetRemover
from gen.pipelines.i23d_triposr import TripoSRImageTo3D
from gen.validators.external_validator import ExternalValidator


class MinerState:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Devices (A40 x2)
        if torch.cuda.is_available():
            self.t2i_device = torch.device(f"cuda:{cfg.t2i_gpu_id}")  # GPU0
            self.aux_device = torch.device(f"cuda:{cfg.aux_gpu_id}")  # GPU1
        else:
            self.t2i_device = torch.device("cpu")
            self.aux_device = torch.device("cpu")

        # Pipelines pinned to devices
        self.t2i = SD35Text2Image(self.t2i_device, self.cfg.sd35_model_id)  # GPU0
        self.bg_remover = BiRefNetRemover(self.aux_device)  # GPU1
        self.triposr_img = TripoSRImageTo3D(  # GPU1
            self.aux_device,
        )

        # External validator (your validator may use GPU0 internally)
        self.validator = ExternalValidator(
            cfg.validator_url_txt, cfg.validator_url_img, cfg.vld_threshold
        )

        self.queue_maxsize: int = getattr(cfg, "queue_maxsize", 4)

        # Knobs
        self.t2i_max_tries: int = getattr(cfg, "t2i_max_tries", 3)
        self.triposr_max_tries: int = getattr(cfg, "triposr_max_tries", 1)
        self.early_stop_score: float = getattr(
            cfg, "early_stop_score", max(0.0, cfg.vld_threshold)
        )
        self.time_budget_s: Optional[float] = getattr(cfg, "time_budget_s", None)

        self.debug_save: bool = bool(getattr(cfg, "debug_save", False))

        logger.info(
            f"Models loaded. [GPU0] SD3.5 (+Validator) on {self.t2i_device}, "
            f"[GPU1] BiRefNet(BG) + TripoSR on {self.aux_device}."
        )

    # ---------------------------
    # Helpers
    # ---------------------------

    async def _run_blocking(self, fn, *args, **kwargs):
        return await asyncio.to_thread(functools.partial(fn, *args, **kwargs))

    def _log_cuda(self, message: str, device: torch.device):
        logger.warning(f"[{device}] {message}")
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def _t2i_param_sweep(self) -> List[Dict]:
        base_steps = self.cfg.sd35_steps
        base_guidance = self.cfg.sd35_cfg
        base_res = self.cfg.sd35_res

        tries: List[Dict] = []
        for i in range(self.t2i_max_tries):
            steps = max(1, base_steps + (i % 2))  # 2/3 steps alternation
            tries.append(
                {
                    "steps": steps,
                    "guidance": base_guidance,
                    "res": base_res,
                    "seed": random.randint(0, 2**31 - 1),
                }
            )
        return tries

    def _triposr_param_sweep(self) -> List[Dict]:
        params = []
        for _ in range(max(1, self.triposr_max_tries)):
            params.append(
                {
                    "seed": random.randint(0, 2**31 - 1),
                }
            )
        return params

    async def _gen_one_image(self, prompt: str, params: dict):
        try:
            img = await self.t2i.generate_async(
                prompt=prompt,
                steps=params["steps"],
                guidance=params["guidance"],
                res=params["res"],
                seed=params["seed"],
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("[GPU0] SD3.5 OOM; clearing cache and skipping one try.")
                if self.t2i_device.type == "cuda":
                    with torch.cuda.device(self.t2i_device):
                        torch.cuda.empty_cache()
                return None, params
            raise
        return img, params

    async def _bg_remove_one(self, pil_image: Image.Image) -> Image.Image | None:
        try:
            fg, _ = await self._run_blocking(self.bg_remover.remove, pil_image)
            return fg
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("[GPU1] BiRefNet OOM; clearing cache and dropping item.")
                if self.t2i_device.type == "cuda":
                    with torch.cuda.device(self.t2i_device):
                        torch.cuda.empty_cache()
                return None
            raise

    async def _triposr_one(self, pil_image, params: dict):
        try:
            ply_bytes = await self._run_blocking(
                self.triposr_img.infer_to_ply,
                pil_image,
                seed=params.get("seed"),
            )
            return ply_bytes, params
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg:
                self._log_cuda("[GPU1] TripoSR OOM; drop this try.", self.aux_device)
                return b"", params
            # Unknown runtime — return empty to keep the pipeline flowing
            self._log_cuda(f"[GPU1] TripoSR runtime error: {e}", self.aux_device)
            return b"", params

    def _within_budget(self, start_ts: float) -> bool:
        if self.time_budget_s is None:
            return True
        return (time.time() - start_ts) < self.time_budget_s

    # -------------------------------------------------
    # Process 1 (GPU0): SD35 + BG  ==> q01
    # -------------------------------------------------
    async def _proc1_t2i_bg(self, prompt, q01, start_ts, stop_evt):
        tasks: List[asyncio.Task] = []

        async def _one(params: dict):
            if stop_evt.is_set() or not self._within_budget(start_ts):
                return
            t0 = time.time()
            img, iparams = await self._gen_one_image(prompt, params)
            t2i_sec = time.time() - t0
            if img is None:
                return

            logger.debug(f"[GPU1/Proc1] T2I {t2i_sec:.2f}s -> q01")
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
    # Process 2 (GPU1): TripoSR   q01 ==> q12
    # -------------------------------------------------
    async def _proc2_triposr(self, q01, q12, start_ts, stop_evt):
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

            if stop_evt.is_set() or not self._within_budget(start_ts):
                break
            tparams = self._triposr_param_sweep()[0]
            t0 = time.time()
            ply_bytes, _ = await self._triposr_one(fg, tparams)
            TripoSR_sec = time.time() - t0

            if not ply_bytes:
                logger.debug("[GPU1/Proc2] Empty PLY; dropping.")
                continue

            logger.debug(
                f"[GPU1/Proc2] TripoSR {TripoSR_sec:.2f}s + BG {bg_sec:.2f}s -> q12"
            )
            # Attach metadata for logging/inspection
            await q12.put((ply_bytes, {"iparams": iparams, "tparams": tparams}))

    # -------------------------------------------------
    # Process 3 (GPU0): Validate  q12
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
            tparams = meta.get("tparams", {})

            v0 = time.time()
            # If your validator uses GPU0, this now runs with GPU0 free of Flux/BG.
            score, passed, _ = await self.validator.validate_text(prompt, ply_bytes)

            vsec = time.time() - v0
            logger.info(
                f"[Proc3/validate] T2I{iparams}|TRIPOSR{tparams} -> score={score:.4f}, "
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
                # We won’t cancel TripoSR actively; queue will drain to sentinel.
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
          Proc1 [GPU0]: T2I+BG -> q01 (signals prod_done when finished)
          Proc2 [GPU1]: TripoSR consumes q01 -> q12
          Proc3 [GPU0]: waits for prod_done, then validates items from q12
        """
        start_ts = time.time()
        q01: asyncio.Queue = asyncio.Queue(maxsize=self.queue_maxsize)
        q12: asyncio.Queue = asyncio.Queue(maxsize=self.queue_maxsize)
        stop_evt = asyncio.Event()
        best_out: Dict[str, object] = {}

        proc1 = asyncio.create_task(self._proc1_t2i_bg(prompt, q01, start_ts, stop_evt))
        proc2 = asyncio.create_task(self._proc2_triposr(q01, q12, start_ts, stop_evt))
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

        # BG removal on GPU0
        fg, _ = self.bg_remover.remove(pil_image)

        best_score = -1.0
        best_ply: bytes = b""

        for tparams in self._triposr_param_sweep():
            if not self._within_budget(start_ts):
                logger.warning("Budget exhausted mid-TripoSR(image); stopping.")
                break

            ply_bytes, _ = await self._triposr_one(fg, tparams)
            if not ply_bytes:
                continue

            score, passed, _ = await self.validator.validate_image(image_b64, ply_bytes)
            logger.info(
                f"[image] TripoSR{tparams} -> score={score:.4f}, passed={passed}"
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
