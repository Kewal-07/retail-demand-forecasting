"""Load every booster once at import time. See CLAUDE.md Section 9/10/15.

Feature order mismatch fails at startup, not at runtime -- a mismatch
here means the saved boosters were trained against a different feature
schema than the currently-checked-out src/features/pipeline.py produces,
which would silently corrupt every prediction if not caught immediately.
"""
import json
from pathlib import Path

import lightgbm as lgb

from src.features.pipeline import FEATURE_COLUMNS

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
QUANTILE_ALPHAS = [0.10, 0.25, 0.50, 0.75, 0.90]


def load_models() -> dict:
    feature_list_path = MODELS_DIR / "feature_list.json"
    saved_feature_list = json.loads(feature_list_path.read_text())
    if saved_feature_list != FEATURE_COLUMNS:
        raise RuntimeError(
            "feature_list.json does not match the live feature pipeline's output "
            "order -- retrain or regenerate feature_list.json before serving"
        )

    models = {"point": lgb.Booster(model_file=str(MODELS_DIR / "point_tweedie_global.txt"))}
    for alpha in QUANTILE_ALPHAS:
        name = f"q{int(alpha * 100):02d}"
        models[name] = lgb.Booster(model_file=str(MODELS_DIR / f"{name}.txt"))
    return models
