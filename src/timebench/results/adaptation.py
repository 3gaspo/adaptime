"""Comparison tables over independently evaluated Adaptime wrappers."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from timebench.pipeline.runs import load_manifest, select_completed_runs


SUPPORT_FIELDS = (
    "dataset_config",
    "num_series",
    "num_windows",
    "num_variates",
    "prediction_length",
    "seasonality",
    "target_mode",
)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _runs(
    root: Path,
    model: str,
    *,
    config_policy: str,
    repeat_policy: str,
) -> list[tuple[Path, dict[str, object]]]:
    if (root / "manifest.json").is_file() and (
        root / "metrics_summary.json"
    ).is_file():
        manifest = load_manifest(root)
        if (
            manifest["status"] != "completed"
            or manifest["identity"]["model"] != model
        ):
            raise FileNotFoundError(f"no completed {model} evaluation at {root}")
        selected = [(root, manifest)]
    else:
        selected = select_completed_runs(
            root,
            models={model},
            config_policy=config_policy,
            repeat_policy=repeat_policy,
        )
    if not selected:
        raise FileNotFoundError(f"no completed {model} evaluation below {root}")
    return selected


def _mean(values: Iterable[object]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return fmean(finite) if finite else None


def _geometric_mean(values: Iterable[object]) -> float:
    finite = [float(value) for value in values]
    if not finite or any(not math.isfinite(value) or value < 0 for value in finite):
        raise ValueError("scaled MASE requires finite non-negative task values")
    if any(value == 0 for value in finite):
        return 0.0
    return math.exp(fmean(math.log(value) for value in finite))


def _combine_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted(
        {name for cell in cells for name in dict(cell["metrics"])}
    )
    metrics: dict[str, dict[str, float | int | None]] = {}
    for metric in metric_names:
        values = [dict(cell["metrics"]).get(metric, {}) for cell in cells]
        metrics[metric] = {
            "mean": _mean(dict(value).get("mean") for value in values),
            "finite_values": sum(
                int(dict(value).get("finite_values", 0)) for value in values
            ),
            "total_values": sum(
                int(dict(value).get("total_values", 0)) for value in values
            ),
        }
    seconds = [cell["inference_seconds"] for cell in cells]
    fallback_reasons = {
        json.dumps(cell["fallback_reason"], sort_keys=True)
        for cell in cells
    }
    if len(fallback_reasons) != 1:
        raise ValueError("selected repeats or configurations disagree on task fallback")
    return {
        "dataset": cells[0]["dataset"],
        "term": cells[0]["term"],
        "method": cells[0]["method"],
        "base_method": cells[0]["base_method"],
        "scientific_config": cells[0]["scientific_config"],
        "support": cells[0]["support"],
        "metrics": metrics,
        "inference_seconds": (
            _mean(seconds) if all(value is not None for value in seconds) else None
        ),
        "fallback_reason": cells[0]["fallback_reason"],
        "manifests": [path for cell in cells for path in cell["manifests"]],
    }


def _effective_cells(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average selected repeats, then selected configurations, per report policy."""

    by_exact_config: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(
        list
    )
    for cell in cells:
        key = (
            cell["dataset"],
            cell["term"],
            cell["method"],
            cell["base_method"],
            json.dumps(cell["scientific_config"], sort_keys=True),
        )
        by_exact_config[key].append(cell)

    config_means = [_combine_cells(group) for group in by_exact_config.values()]
    by_task_method: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for cell in config_means:
        key = (
            cell["dataset"],
            cell["term"],
            cell["method"],
            cell["base_method"],
        )
        by_task_method[key].append(cell)
    return [_combine_cells(group) for group in by_task_method.values()]


def build_adaptation_comparison(
    *,
    method_results_roots: dict[str, str | Path],
    output_dir: str | Path,
    expected_tasks: Iterable[tuple[str, str]],
    config_policy: str = "error",
    repeat_policy: str = "selected",
) -> Path:
    """Join task results and produce foundation-compatible scaled MASE."""

    roots = {
        method: Path(results_root).expanduser().resolve()
        for method, results_root in method_results_roots.items()
    }
    if not roots:
        raise ValueError("comparison requires at least one method-results root")
    expected = set(expected_tasks)
    cells: list[dict[str, Any]] = []
    for base_method, root in roots.items():
        for run_dir, manifest in _runs(
            root,
            base_method,
            config_policy=config_policy,
            repeat_policy=repeat_policy,
        ):
            identity = dict(manifest["identity"])
            key = (str(identity["dataset"]), str(identity["term"]))
            summary = json.loads(
                (run_dir / "metrics_summary.json").read_text(encoding="utf-8")
            )
            dataset_path = str(summary["dataset_config"]).rsplit("/", 1)[0]
            requested_key = (dataset_path, key[1])
            if requested_key in expected:
                config = json.loads(
                    (run_dir / "config.json").read_text(encoding="utf-8")
                )
                selection = dict(manifest.get("selection", {}))
                cells.append(
                    {
                        "dataset": requested_key[0],
                        "term": requested_key[1],
                        "method": selection.get("model_label", base_method),
                        "base_method": base_method,
                        "scientific_config": selection.get(
                            "scientific_config",
                            {
                                "model_config": manifest.get("model_config", {}),
                                "pipeline_config": manifest.get("pipeline_config", {}),
                                "experiment_config": manifest.get("experiment_config", {}),
                            },
                        ),
                        "support": {
                            field: config.get(field) for field in SUPPORT_FIELDS
                        },
                        "metrics": dict(summary["metrics"]),
                        "inference_seconds": summary.get("inference_seconds"),
                        "fallback_reason": config.get("adaptation_fallback_reason"),
                        "manifests": [str(run_dir / "manifest.json")],
                    }
                )

    missing = [
        key
        for key in sorted(expected)
        if {
            cell["base_method"]
            for cell in cells
            if (cell["dataset"], cell["term"]) == key
        }
        != set(roots)
    ]
    if missing:
        raise FileNotFoundError(f"comparison is missing independently evaluated tasks: {missing}")

    for dataset, term in sorted(expected):
        task_cells = [
            cell
            for cell in cells
            if (cell["dataset"], cell["term"]) == (dataset, term)
        ]
        support = {
            f"{cell['method']}:{index}": cell["support"]
            for index, cell in enumerate(task_cells)
        }
        if len({json.dumps(value, sort_keys=True) for value in support.values()}) != 1:
            raise ValueError(
                f"method evaluation support differs for {dataset}/{term}: "
                f"{support}"
            )

    effective = _effective_cells(cells)
    baseline_by_task: dict[tuple[str, str], float] = {}
    for cell in effective:
        if cell["base_method"] != "seasonal_naive":
            continue
        key = (cell["dataset"], cell["term"])
        mase = dict(cell["metrics"]).get("MASE", {}).get("mean")
        if mase is None or not math.isfinite(float(mase)) or float(mase) <= 0:
            raise ValueError(
                f"missing positive Seasonal Naive MASE for {cell['dataset']}/{cell['term']}"
            )
        if key in baseline_by_task:
            raise ValueError(f"multiple Seasonal Naive baselines for {cell['dataset']}/{cell['term']}")
        baseline_by_task[key] = float(mase)
    missing_baselines = sorted(expected - set(baseline_by_task))
    if missing_baselines:
        raise FileNotFoundError(
            f"comparison is missing Seasonal Naive MASE baselines: {missing_baselines}"
        )
    for cell in effective:
        mase = dict(cell["metrics"]).get("MASE", {}).get("mean")
        if mase is None or not math.isfinite(float(mase)) or float(mase) < 0:
            raise ValueError(
                f"missing finite non-negative MASE for {cell['method']} "
                f"on {cell['dataset']}/{cell['term']}"
            )
        cell["scaled_MASE"] = float(mase) / baseline_by_task[
            (cell["dataset"], cell["term"])
        ]
    metric_names = sorted(
        {name for cell in effective for name in dict(cell["metrics"])}
    )
    rows: list[dict[str, object]] = []
    for cell in sorted(
        effective,
        key=lambda value: (
            value["dataset"], value["term"], value["base_method"], value["method"]
        ),
    ):
        row: dict[str, object] = {
            "dataset": cell["dataset"],
            "term": cell["term"],
            "method": cell["method"],
            "base_method": cell["base_method"],
            "inference_seconds": cell["inference_seconds"],
            "task_fallback": cell["fallback_reason"] is not None,
            "task_fallback_reason": (
                ""
                if cell["fallback_reason"] is None
                else json.dumps(cell["fallback_reason"], sort_keys=True)
            ),
            "scaled_MASE": cell["scaled_MASE"],
        }
        for metric in metric_names:
            values = dict(cell["metrics"]).get(metric, {})
            row[metric] = values.get("mean")
            row[f"{metric}_finite_values"] = values.get("finite_values", 0)
            row[f"{metric}_total_values"] = values.get("total_values", 0)
        rows.append(row)

    by_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in effective:
        by_method[(cell["base_method"], cell["method"])].append(cell)
    summary_rows: list[dict[str, object]] = []
    for (base_method, method), method_cells in by_method.items():
        timed = [
            float(cell["inference_seconds"])
            for cell in method_cells
            if cell["inference_seconds"] is not None
            and math.isfinite(float(cell["inference_seconds"]))
        ]
        summary_rows.append(
            {
                "model": method,
                "base_model": base_method,
                "target_modes": ",".join(
                    sorted(
                        {
                            str(cell["support"].get("target_mode"))
                            for cell in method_cells
                        }
                    )
                ),
                "scaled_MASE": _geometric_mean(
                    cell["scaled_MASE"] for cell in method_cells
                ),
                "inference_seconds": (
                    sum(timed) if len(timed) == len(method_cells) else None
                ),
                "datasets": len({cell["dataset"] for cell in method_cells}),
                "tasks": len(method_cells),
                "timed_tasks": len(timed),
                "MASE_finite_values": sum(
                    int(dict(cell["metrics"])["MASE"].get("finite_values", 0))
                    for cell in method_cells
                ),
                "MASE_total_values": sum(
                    int(dict(cell["metrics"])["MASE"].get("total_values", 0))
                    for cell in method_cells
                ),
            }
        )
    summary_rows.sort(key=lambda row: (float(row["scaled_MASE"]), str(row["model"])))

    root = Path(output_dir).expanduser().resolve()
    _atomic_csv(root / "comparison.csv", rows)
    _atomic_csv(root / "adaptation_summary.csv", summary_rows)
    manifest_path = root / "report_manifest.json"
    _atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "format": "adaptime_independent_evaluation_comparison",
            "status": "completed",
            "method_results_roots": {
                method: str(results_root)
                for method, results_root in roots.items()
            },
            "selection": {
                "config_policy": config_policy,
                "repeat_policy": repeat_policy,
            },
            "metric_columns": {
                "mean": "<metric>",
                "scaled_MASE": "task MASE divided by matching Seasonal Naive MASE",
                "finite_values": "<metric>_finite_values",
                "total_values": "<metric>_total_values",
            },
            "aggregation": {
                "scaled_MASE": "geometric_mean_over_tasks",
                "inference_seconds": "sum_over_tasks_when_all_are_timed",
            },
            "fallback_columns": {
                "task_fallback": "whether this method used its task-level fallback",
                "task_fallback_reason": "recorded fallback reason or an empty string",
            },
            "input_manifests": [
                path for cell in cells for path in cell["manifests"]
            ],
            "files": {
                "comparison": "comparison.csv",
                "summary": "adaptation_summary.csv",
            },
        },
    )
    return manifest_path
