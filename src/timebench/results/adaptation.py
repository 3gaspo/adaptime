"""Comparison tables over independently evaluated Adaptime wrappers."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterable

from timebench.pipeline.runs import select_completed_runs


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
    if (root / "manifest.json").is_file() and (root / "metrics_summary.json").is_file():
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
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


def build_adaptation_comparison(
    *,
    method_results_roots: dict[str, str | Path],
    output_dir: str | Path,
    expected_tasks: Iterable[tuple[str, str]],
    config_policy: str = "error",
    repeat_policy: str = "selected",
) -> Path:
    """Join independently evaluated method summaries on identical support."""

    roots = {
        method: Path(results_root).expanduser().resolve()
        for method, results_root in method_results_roots.items()
    }
    if not roots:
        raise ValueError("comparison requires at least one method-results root")
    expected = set(expected_tasks)
    by_task: dict[tuple[str, str], dict[str, tuple[Path, dict[str, object]]]] = {}
    for model, root in roots.items():
        for run_dir, manifest in _runs(
            root,
            model,
            config_policy=config_policy,
            repeat_policy=repeat_policy,
        ):
            identity = dict(manifest["identity"])
            key = (str(identity["dataset"]), str(identity["term"]))
            dataset_path = str(
                json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))[
                    "dataset_config"
                ]
            ).rsplit("/", 1)[0]
            requested_key = (dataset_path, key[1])
            if requested_key in expected:
                by_task.setdefault(requested_key, {})[model] = (run_dir, manifest)

    missing = [key for key in sorted(expected) if set(by_task.get(key, {})) != set(roots)]
    if missing:
        raise FileNotFoundError(f"comparison is missing independently evaluated tasks: {missing}")

    rows: list[dict[str, object]] = []
    manifests: list[str] = []
    for (dataset, term), methods in sorted(by_task.items()):
        comparable = {
            model: json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            for model, (run_dir, _) in methods.items()
        }
        support_fields = (
            "dataset_config",
            "num_series",
            "num_windows",
            "num_variates",
            "prediction_length",
            "seasonality",
            "target_mode",
        )
        support = {
            model: {field: config.get(field) for field in support_fields}
            for model, config in comparable.items()
        }
        if len({json.dumps(value, sort_keys=True) for value in support.values()}) != 1:
            raise ValueError(
                f"method evaluation support differs for {dataset}/{term}: "
                f"{support}"
            )
        for model in roots:
            run_dir, _ = methods[model]
            summary = json.loads(
                (run_dir / "metrics_summary.json").read_text(encoding="utf-8")
            )
            row: dict[str, object] = {
                "dataset": dataset,
                "term": term,
                "method": model,
                "inference_seconds": float(summary["inference_seconds"]),
            }
            for metric, values in dict(summary["metrics"]).items():
                row[str(metric)] = values["mean"]
            rows.append(row)
            manifests.append(str(run_dir / "manifest.json"))

    root = Path(output_dir).expanduser().resolve()
    _atomic_csv(root / "comparison.csv", rows)
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
            "input_manifests": manifests,
            "files": {"comparison": "comparison.csv"},
        },
    )
    return manifest_path
