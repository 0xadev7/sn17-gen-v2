from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    port: int = int(os.getenv("PORT", 7000))
    timeout_s: float = float(os.getenv("TIMEOUT_S", 30.0))

    # Validation
    validator_url_txt: str = os.getenv(
        "VALIDATOR_TXT_URL", "http://localhost:8094/validate_txt_to_3d_ply/"
    )
    validator_url_img: str = os.getenv(
        "VALIDATOR_IMG_URL", "http://localhost:8094/validate_img_to_3d_ply/"
    )
    vld_threshold: float = float(os.getenv("VALIDATION_THRESHOLD", 0.7))

    # Early stop & budget
    early_stop_score: float = float(os.getenv("EARLY_STOP_SCORE", 0.82))
    time_budget_s: float | None = float(os.getenv("TIME_BUDGET_S", 22))

    # Text-to-2D parameters
    t2i_steps: int = int(os.getenv("T2I_STEPS", 4))
    t2i_guidance: float = float(os.getenv("T2I_GUIDANCE", 0.0))
    t2i_res: int = int(os.getenv("T2I_RES", 1024))
    t2i_max_tries: int = int(os.getenv("T2I_MAX_TRIES", 1))

    # Trellis parameters
    trellis_struct_steps: int = int(os.getenv("TRELLIS_STRUCT_STEPS", 8))
    trellis_slat_steps: int = int(os.getenv("TRELLIS_SLAT_STEPS", 10))
    trellis_cfg_struct: float = float(os.getenv("TRELLIS_CFG_STRUCT", 7.5))
    trellis_cfg_slat: float = float(os.getenv("TRELLIS_CFG_SLAT", 3.0))
    trellis_max_tries: int = int(os.getenv("TRELLIS_MAX_TRIES", 1))

    # Multi-view (Zero123) parameters
    mv_num_views: int = int(os.getenv("MV_NUM_VIEWS", 8))
    mv_res: int = int(os.getenv("MV_RES", 768))
    # Comma-separated yaw degrees; if empty, we’ll auto spread over 360°
    mv_yaws_csv: str = os.getenv("MV_YAWS", "")
    # How many top views to try in Trellis (sorted by selector score)
    mv_topk_for_trellis: int = int(os.getenv("MV_TOPK_FOR_TRELLIS", 4))

    # Save intermediary results
    debug_save: bool = os.getenv("DEBUG_SAVE", "0") == "1"
