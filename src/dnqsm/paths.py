"""Configurable local research-data defaults; no dependency on whnet."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = Path(
    os.environ.get("DNQSM_REFERENCE_ROOT", PROJECT_ROOT.parent / "wh_net")
)
TRAINING_DATA_DIR = REFERENCE_ROOT / "data/training/joint"
EVALUATION_DATA_DIR = REFERENCE_ROOT / "data/evaluation"
RUNS_DIR = PROJECT_ROOT / "artifacts/runs"
TRAINING_CACHE_DIR = PROJECT_ROOT / "artifacts/cache/training"
