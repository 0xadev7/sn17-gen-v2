from __future__ import annotations
import os
from dataclasses import dataclass


def _getenv(k: str, default: str) -> str:
    v = os.getenv(k)
    return v if v not in (None, "") else default


def _getfloat(k: str, default: float) -> float:
    try:
        return float(_getenv(k, str(default)))
    except Exception:
        return default


def _getint(k: str, default: int) -> int:
    try:
        return int(_getenv(k, str(default)))
    except Exception:
        return default


@dataclass
class Config:
    # Server
    port: int = _getint("PORT", 7000)
    timeout_s: float = _getfloat("TIMEOUT_S", 30.0)

    # GPU assignment (supports single or dual GPUs)
    t2i_gpu_id: int = _getint("T2I_GPU_ID", 0)
    aux_gpu_id: int = _getint("AUX_GPU_ID", 0)

    # Validator
    validator_url_txt: str = _getenv(
        "VALIDATOR_URL_TXT", "http://127.0.0.1:9000/validate_text/"
    )
    validator_url_img: str = _getenv(
        "VALIDATOR_URL_IMG", "http://127.0.0.1:9000/validate_image/"
    )
    vld_threshold: float = _getfloat("VALIDATOR_THRESHOLD", 0.70)

    # Early stop & budget
    early_stop_score: float = _getfloat("EARLY_STOP_SCORE", 0.85)
    time_budget_s: float = _getfloat("TIME_BUDGET_S", 28.5)  # leave a little headroom

    # Concurrency & buffering
    queue_maxsize: int = _getint("QUEUE_MAXSIZE", 4)
    t2i_concurrency: int = _getint("T2I_CONCURRENCY", 1)
    triposr_concurrency: int = _getint("TRIPOSR_CONCURRENCY", 1)

    # T2I (SD3.5) fast presets
    sd35_model_id: str = _getenv(
        "SD35_MODEL_ID", "stabilityai/stable-diffusion-3.5-large"
    )
    sd35_res: int = _getint("SD35_RES", 576)
    sd35_steps: int = _getint("SD35_STEPS", 18)
    sd35_cfg: float = _getfloat("SD35_CFG", 3.5)
    sd35_max_tries: int = _getint("SD35_MAX_TRIES", 3)
    sd35_enable_xformers: bool = _getenv("SD35_XFORMERS", "1") == "1"

    # TripoSR parameters
    triposr_model_id: str = _getenv("TRIPOSR_MODEL_ID", "stabilityai/TripoSR")
    triposr_chunk_size: int = _getint("TRIPOSR_CHUNK_SIZE", 8192)
    triposr_max_tries: int = int(os.getenv("TRIPOSR_MAX_TRIES", 1))

    # Debug
    debug_save: bool = os.getenv("DEBUG_SAVE", "0") == "1"
