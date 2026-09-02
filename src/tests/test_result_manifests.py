"""Focused manifest-layout and TIME-reader integration check."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import tempfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _task(
    root: Path,
    *,
    context_length: int,
    mase: float,
    term: str = "short",
    complete: bool = True,
) -> Path:
    from timebench.pipeline import allocate_run

    identity_root = (
        root
        / "expe_uni"
        / "chronos2"
        / "multivariate"
        / "toy"
        / "H"
        / term
    )
    handle = allocate_run(
        identity_root,
        experiment="expe_uni",
        identity={
            "model": "chronos2",
            "target_mode": "multivariate",
            "dataset": "toy",
            "frequency": "H",
            "term": term,
        },
        model_config={
            "model_size": "chronos2",
            "context_length": context_length,
            "quantile_levels": [0.5],
        },
        pipeline_config={
            "prediction_length": 2,
            "test_length": 4,
            "val_length": 0,
            "windows": 2,
        },
        runtime_config={"batch_size": 1, "device": "cpu"},
        experiment_config={"covariate_mode": "none"},
    )
    if not complete:
        try:
            with handle:
                raise RuntimeError("synthetic failure")
        except RuntimeError:
            pass
        return handle.run_dir

    with handle:
        (handle.run_dir / "config.json").write_text(
            json.dumps({"inference_seconds": 1.25}), encoding="utf-8"
        )
        (handle.run_dir / "metrics_summary.json").write_text(
            json.dumps({"metrics": {"MASE": {"mean": mase}}}),
            encoding="utf-8",
        )
        np.savez(handle.run_dir / "metrics.npz", MASE=np.array([mase]), CRPS=np.array([mase]), MAE=np.array([mase]), MSE=np.array([mase]))
        np.savez(handle.run_dir / "predictions.npz", predictions=np.array([mase]))
        handle.complete(
            ["predictions.npz", "metrics.npz", "config.json", "metrics_summary.json"]
        )
    return handle.run_dir


def main() -> None:
    from timebench.paths import foundation_experiment_name, foundation_identity_root
    from timebench.pipeline import ManifestError, load_manifest, select_completed_runs

    summary = _load_module(
        "timebench_foundation_summary",
        PROJECT_ROOT / "scripts/compute_foundation_summary.py",
    )
    leaderboard = _load_module(
        "timebench_local_leaderboard",
        PROJECT_ROOT / "scripts/compute_local_leaderboard.py",
    )

    for path in [
        *sorted((PROJECT_ROOT / "experiments").glob("*.py")),
        PROJECT_ROOT / "src/timebench/pipeline/runs.py",
        PROJECT_ROOT / "scripts/compute_foundation_summary.py",
        PROJECT_ROOT / "scripts/compute_local_leaderboard.py",
    ]:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    assert foundation_experiment_name("none") == "expe_uni"
    assert foundation_experiment_name("future_included") == "expe_covar"
    assert foundation_experiment_name("past_targets") == "expe_covar"

    previous_launch = os.environ.get("TIME_LAUNCH_ID")
    os.environ["TIME_LAUNCH_ID"] = "launch_1"
    try:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as directory:
            root = Path(directory)
            first = _task(root, context_length=2048, mase=2.0)
            second = _task(root, context_length=2048, mase=1.0)
            assert first.name == "run_0" and second.name == "run_1"
            expected_root = foundation_identity_root(
                root / "expe_uni",
                "chronos2",
                "multivariate",
                "toy/H",
                "short",
            )
            assert second.parent == expected_root

            selected = select_completed_runs(root / "expe_uni")
            assert [path.name for path, _ in selected] == ["run_1"]
            cells = summary.load_result_cells(root / "expe_uni")
            assert len(cells) == 1 and cells[0]["MASE"] == 1.0

            failed = _task(
                root,
                context_length=2048,
                mase=9.0,
                term="medium",
                complete=False,
            )
            assert load_manifest(failed)["status"] == "interrupted"

            latest_config = _task(root, context_length=1024, mase=3.0)
            try:
                select_completed_runs(root / "expe_uni")
            except ManifestError as error:
                assert "Multiple scientific run configurations" in str(error)
            else:
                raise AssertionError("Different scientific configs must not mix silently")

            filtered = summary.load_result_cells(
                root / "expe_uni",
                config_filters={"model_config.context_length": 2048},
            )
            assert len(filtered) == 1 and filtered[0]["MASE"] == 1.0
            latest = summary.load_result_cells(
                root / "expe_uni", config_policy="latest"
            )
            assert len(latest) == 1 and latest[0]["MASE"] == 3.0
            assert latest[0]["manifest_path"] == str(latest_config / "manifest.json")

            frame = leaderboard.get_manifest_datasets_results(
                root / "expe_uni",
                config_filters={"model_config.context_length": 2048},
            )
            assert frame.iloc[0]["model"] == "chronos2"
            assert frame.iloc[0]["MASE"] == 1.0
    finally:
        if previous_launch is None:
            os.environ.pop("TIME_LAUNCH_ID", None)
        else:
            os.environ["TIME_LAUNCH_ID"] = previous_launch

    source = (PROJECT_ROOT / "src/timebench/pipeline/runs.py").read_text(
        encoding="utf-8"
    )
    assert "hashlib" not in source
    print("Manifest layout and TIME result readers passed")


if __name__ == "__main__":
    main()
