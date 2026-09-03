"""Focused dependency-light contract check for TIME timing and aggregation."""

import importlib.util
import json
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


def main() -> None:
    timing = _load_module(
        "timebench_timing_contract",
        PROJECT_ROOT / "src/timebench/evaluation/timing.py",
    )
    synchronizations = []
    timing._synchronize_accelerator = lambda: synchronizations.append(True)
    timer = timing.EvaluationTimer()
    timer.start()
    assert timer.stop() >= 0.0
    assert len(synchronizations) == 2

    summary = _load_module(
        "timebench_foundation_summary",
        PROJECT_ROOT / "scripts/compute_foundation_summary.py",
    )
    evaluation_utils = _load_module(
        "timebench_evaluation_utils",
        PROJECT_ROOT / "src/timebench/evaluation/utils.py",
    )

    tensor_quantiles = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    normalized_tensor = evaluation_utils.normalize_tsicl_quantiles(tensor_quantiles)
    assert normalized_tensor.shape == (2, 5, 3, 4)
    assert np.array_equal(normalized_tensor, tensor_quantiles.transpose(0, 3, 1, 2))

    list_quantiles = [
        np.arange(3 * 4 * 5).reshape(3, 4, 5),
        np.arange(3 * 4 * 5, 2 * 3 * 4 * 5).reshape(3, 4, 5),
    ]
    normalized_list = evaluation_utils.normalize_tsicl_quantiles(list_quantiles)
    assert normalized_list.shape == (2, 5, 3, 4)
    assert np.array_equal(normalized_list[0], list_quantiles[0].transpose(2, 0, 1))

    with tempfile.TemporaryDirectory() as temporary:
        result_root = Path(temporary) / "results"
        task_dir = result_root / "expe_uni/model_a/univariate/dataset_1/H/short/run_0"
        task_dir.mkdir(parents=True)
        (task_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "project": "improved",
                    "experiment": "expe_uni",
                    "identity": {
                        "model": "model_a",
                        "target_mode": "univariate",
                        "dataset": "dataset_1",
                        "frequency": "H",
                        "term": "short",
                    },
                    "model_config": {},
                    "pipeline_config": {},
                    "runtime_config": {},
                    "experiment_config": {"covariate_mode": "none"},
                    "launch": {"launch_id": "launch_1", "slurm_job_id": None},
                    "status": "completed",
                    "required_artifacts": ["config.json", "metrics_summary.json"],
                }
            ),
            encoding="utf-8",
        )
        (task_dir / "config.json").write_text(
            json.dumps({"launch_id": "launch_1", "inference_seconds": 1.25}),
            encoding="utf-8",
        )
        (task_dir / "metrics_summary.json").write_text(
            json.dumps({"metrics": {"MASE": {"mean": 2.5}}}),
            encoding="utf-8",
        )
        cells = summary.load_result_cells(
            result_root / "expe_uni",
            {"model_a"},
            launch_id="launch_1",
        )
        assert cells == [
            {
                "model": "model_a",
                "target_mode": "univariate",
                "dataset_id": "dataset_1/H",
                "horizon": "short",
                "MASE": 2.5,
                "inference_seconds": 1.25,
                "manifest_path": str(task_dir / "manifest.json"),
            }
        ]

        status_dir = Path(temporary) / "status"
        status_dir.mkdir()
        (status_dir / "model_a.status").write_text(
            "launch_id=launch_1\nstate=completed\nexit_code=0\n",
            encoding="utf-8",
        )
        (status_dir / "model_b.status").write_text(
            "launch_id=launch_1\nstate=failed\nexit_code=1\n",
            encoding="utf-8",
        )
        statuses = summary.load_model_statuses(status_dir)
        status_rows = summary.add_model_status(
            summary.summarize_cells(cells),
            ["model_a", "model_b"],
            statuses,
            "launch_1",
        )
        by_status_model = {row["model"]: row for row in status_rows}
        assert by_status_model["model_a"]["state"] == "completed"
        assert by_status_model["model_a"]["tasks"] == 1
        assert by_status_model["model_b"]["state"] == "failed"
        assert by_status_model["model_b"]["tasks"] == 0
    rows = summary.summarize_cells(
        [
            {
                "model": "model_a",
                "target_mode": "univariate",
                "dataset_id": "dataset_1/H",
                "horizon": "short",
                "MASE": 1.0,
                "inference_seconds": 1.0,
            },
            {
                "model": "model_a",
                "target_mode": "multivariate",
                "dataset_id": "dataset_1/H",
                "horizon": "long",
                "MASE": 3.0,
                "inference_seconds": 2.0,
            },
            {
                "model": "model_a",
                "target_mode": "univariate",
                "dataset_id": "dataset_2/D",
                "horizon": "short",
                "MASE": 6.0,
                "inference_seconds": 3.0,
            },
            {
                "model": "model_b",
                "target_mode": "univariate",
                "dataset_id": "dataset_1/H",
                "horizon": "short",
                "MASE": 5.0,
                "inference_seconds": None,
            },
        ]
    )
    by_model = {row["model"]: row for row in rows}
    assert by_model["model_a"]["MASE_macro"] == 4.0
    assert by_model["model_a"]["inference_seconds"] == 6.0
    assert by_model["model_a"]["tasks"] == 3
    assert by_model["model_b"]["inference_seconds"] is None

    runners = sorted(
        path
        for path in (PROJECT_ROOT / "experiments").glob("*.py")
        if path.name != "__init__.py"
    )
    assert [path.name for path in runners] == [
        "chronos2.py",
        "chronos_bolt.py",
        "seasonal_naive.py",
        "ts_icl.py",
    ]
    for runner in runners:
        source = runner.read_text(encoding="utf-8")
        assert "from timebench.evaluation.timing import EvaluationTimer" in source
        assert source.count("timer.start()") == 1
        assert source.count("timer.stop()") == 1
        assert source.count("inference_seconds=inference_seconds") + source.count(
            "inference_seconds = inference_seconds"
        ) == 1

    submit = (PROJECT_ROOT / "scripts/submit_foundation_models.sh").read_text(
        encoding="utf-8"
    )
    assert 'dependency="afterany:$dependency"' in submit
    publisher = (PROJECT_ROOT / "publish_job.sh").read_text(encoding="utf-8")
    result_sync = (PROJECT_ROOT / "sync_results_to_dgx.sh").read_text(
        encoding="utf-8"
    )
    assert "metrics_summary.json" in publisher
    assert "metrics_summary.json" in result_sync
    assert "workflow_status/***" in result_sync

    seasonal = (PROJECT_ROOT / "experiments/seasonal_naive.py").read_text(
        encoding="utf-8"
    )
    assert "warnings.filterwarnings" in seasonal
    assert "Period with BDay freq is deprecated" in seasonal

    print("Timing and foundation-summary contract passed.")


if __name__ == "__main__":
    main()
