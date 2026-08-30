"""Runtime path contract shared by TIME commands and cluster wrappers."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _configured_path(variable: str, fallback: Path) -> Path:
    load_dotenv(PROJECT_ROOT / ".env")
    value = os.getenv(variable)
    return Path(value).expanduser() if value else fallback


def data_root() -> Path:
    """Prepared CSV, summary, and Arrow dataset workspace."""
    return _configured_path("TIME_DATA_ROOT", PROJECT_ROOT / "datasets")


def dataset_storage_root() -> Path:
    """HF Arrow datasets consumed by :class:`timebench.evaluation.Dataset`."""
    return _configured_path("TIME_DATASET", data_root() / "hf_dataset")


def weights_root() -> Path:
    """Model checkpoints and package caches."""
    return _configured_path("TIME_WEIGHTS", PROJECT_ROOT / "weights")


def outputs_root() -> Path:
    """Generated predictions, metrics, features, and reports."""
    return _configured_path("TIME_OUTPUTS", PROJECT_ROOT / "outputs")


def results_root() -> Path:
    """Per-model TIME evaluation results."""
    return outputs_root() / "results"


def logs_root() -> Path:
    """Runtime streams and scheduler logs."""
    return _configured_path("TIME_LOGS", PROJECT_ROOT / "logs")
