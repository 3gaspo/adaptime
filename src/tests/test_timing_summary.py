"""Focused dependency-light contract check for TIME timing and aggregation."""

import importlib.util
from pathlib import Path


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
    rows = summary.summarize_cells(
        [
            {
                "model": "model_a",
                "dataset_id": "dataset_1/H",
                "horizon": "short",
                "MASE": 1.0,
                "inference_seconds": 1.0,
            },
            {
                "model": "model_a",
                "dataset_id": "dataset_1/H",
                "horizon": "long",
                "MASE": 3.0,
                "inference_seconds": 2.0,
            },
            {
                "model": "model_a",
                "dataset_id": "dataset_2/D",
                "horizon": "short",
                "MASE": 6.0,
                "inference_seconds": 3.0,
            },
            {
                "model": "model_b",
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
        "tirex_model.py",
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

    print("Timing and foundation-summary contract passed.")


if __name__ == "__main__":
    main()
