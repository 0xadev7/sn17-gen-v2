from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


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

    # Validator
    validator_url_txt: str = _getenv(
        "VALIDATOR_TXT_URL", "http://localhost:8094/validate_txt_to_3d_ply/"
    )
    validator_url_img: str = _getenv(
        "VALIDATOR_IMG_URL", "http://localhost:8094/validate_img_to_3d_ply/"
    )
    vld_threshold: float = _getfloat("VALIDATOR_THRESHOLD", 0.6)

    # Early stop & budget
    early_stop_score: float = _getfloat("EARLY_STOP_SCORE", 0.9)
    time_budget_s: float = _getfloat("TIME_BUDGET_S", 25.0)

    # T2I (SD3.5) fast presets
    sd35_steps: int = _getint("SD35_STEPS", 4)
    sd35_res: int = _getint("SD35_RES", 1024)
    sd35_max_tries: int = _getint("SD35_MAX_TRIES", 1)
    sd35_enable_xformers: bool = _getenv("SD35_XFORMERS", "1") == "1"

    # I23D (HunYuan) presets
    hunyuan_max_tries: int = _getint("HUNYUAN_MAX_TRIES", 1)

    # Concurrency
    queue_maxsize: int = _getint("QUEUE_MAXSIZE", 4)

    # Debug
    debug_save: bool = os.getenv("DEBUG_SAVE", "0") == "1"
