"""Configurable local research-data defaults; no dependency on whnet."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = Path(
    os.environ.get("DNQSM_REFERENCE_ROOT", PROJECT_ROOT.parent / "wh_net")
)
EVALUATION_DATA_DIR = REFERENCE_ROOT / "data/evaluation"
DEFAULT_MODEL_CHECKPOINT = PROJECT_ROOT / "artifacts/models/current.pt"
