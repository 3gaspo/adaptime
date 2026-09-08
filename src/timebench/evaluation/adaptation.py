"""Common TIME evaluation for forecasts produced by adaptation wrappers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from timebench.evaluation.adaptation_data import PreparedDataset
from timebench.evaluation.data import Dataset
from timebench.evaluation.saver import save_window_predictions
from timebench.pipeline.adaptation_prediction import open_point_predictions


def _storage_root(source_path: Path, dataset_name: str) -> Path:
    root = source_path
    for _ in Path(dataset_name).parts:
        root = root.parent
    return root


def evaluate_point_predictions(
    prepared_path: str | Path,
    prediction_path: str | Path,
    output_dir: str | Path,
    *,
    method: str | None = None,
) -> dict[str, object]:
    """Evaluate one wrapper through the same saver used by foundation models."""

    prepared = PreparedDataset(prepared_path)
    prediction_root, prediction = open_point_predictions(prediction_path)
    if prediction["prepared_signature"] != prepared.signature:
        raise ValueError("predictions and shared TIME data do not match")
    if prepared.target_mode != "univariate":
        raise ValueError("adaptation wrapper evaluation currently requires univariate rows")

    if prediction["format"] == "adaptime_family_predictions":
        available = tuple(prediction["methods"])
        if method not in available:
            raise ValueError(f"method must be one of {available} for this prediction")
        prediction_file = prediction["files"]["predictions"][method]
        inference_seconds = float(prediction["inference_seconds"][method])
        prediction_method = str(method)
    else:
        if method is not None and method != prediction["method"]:
            raise ValueError("requested method does not match point predictions")
        prediction_file = prediction["files"]["predictions"]
        inference_seconds = float(prediction["inference_seconds"])
        prediction_method = str(prediction["method"])
    values = np.load(prediction_root / prediction_file, mmap_mode="r")
    expected = (
        len(prepared.indices("test")),
        1,
        prepared.prediction_length,
    )
    if values.shape != expected:
        raise ValueError(f"point predictions have shape {values.shape}, expected {expected}")

    dataset_name = str(prepared.config["dataset"])
    term = str(prepared.config["term"])
    source_path = Path(prepared.manifest["source_path"]).expanduser().resolve()
    dataset = Dataset(
        dataset_name,
        term=term,
        to_univariate=True,
        prediction_length=prepared.prediction_length,
        test_length=int(prepared.config["test_length"]),
        val_length=0,
        storage_path=_storage_root(source_path, dataset_name),
    )
    expected_rows = dataset.windows * len(dataset.hf_dataset) * dataset.target_dim
    if expected_rows != len(values):
        raise ValueError(
            "shared prepared test rows do not match TIME's foundation evaluator order"
        )

    # A wrapper returns a deterministic point forecast.  Expose it as its
    # median quantile so the standard TIME saver owns shape reconstruction,
    # ground-truth loading, metrics, and the task artifact contract.
    quantiles = np.asarray(values[:, 0, :])[:, None, :]
    return save_window_predictions(
        dataset=dataset,
        fc_quantiles=quantiles,
        ds_config=f"{dataset_name}/{term}",
        output_base_dir=str(Path(output_dir).expanduser().resolve().parent),
        seasonality=prepared.seasonality,
        model_hyperparams={
            "model": prediction_method,
            "experiment": "adaptime",
            "target_mode": prepared.target_mode,
            "forecast_type": "point",
            "prediction_manifest": str(
                (prediction_root / "prediction_manifest.json").resolve()
            ),
            "context_length": int(prediction["context_length"]),
            "selected_adaptation": prediction.get("selected"),
            "bayes_probability_covariate_better": prediction.get(
                "bayes_probability_covariate_better"
            ),
            "adaptation_fallback_reason": prediction.get("fallback_reason"),
        },
        quantile_levels=[0.5],
        inference_seconds=inference_seconds,
        task_output_dir=str(Path(output_dir).expanduser().resolve()),
    )
