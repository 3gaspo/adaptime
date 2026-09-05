#!/usr/bin/env python3
"""Create a macro-MASE and inference-time table from local TIME results."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from timebench.paths import foundation_experiment_root
from timebench.pipeline import parse_config_filters, select_completed_runs

DEFAULT_MODELS = (
    "chronos_bolt",
    "chronos2",
    "ts_icl",
    "seasonal_naive",
)


def load_result_cells(
    root: Path,
    models: set[str] | None = None,
    launch_id: str | None = None,
    target_modes: set[str] | None = None,
    config_filters: dict | None = None,
    config_policy: str = "error",
    repeat_policy: str = "selected",
) -> list[dict]:
    """Load one selected completed manifest per dataset/frequency/horizon cell."""
    cells = []
    selected = select_completed_runs(
        root,
        models=models,
        target_modes=target_modes,
        launch_id=launch_id,
        config_filters=config_filters,
        config_policy=config_policy,
        repeat_policy=repeat_policy,
    )
    for run_dir, manifest in selected:
        identity = manifest["identity"]
        summary_path = run_dir / "metrics_summary.json"
        config_path = summary_path.with_name("config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        mase = summary.get("metrics", {}).get("MASE", {}).get("mean")
        if mase is None:
            continue
        mase = float(mase)
        inference_seconds = config.get("inference_seconds")
        if inference_seconds is not None:
            inference_seconds = float(inference_seconds)
        selection = manifest.get("selection", {})

        cells.append(
            {
                "model": selection.get("model_label", identity["model"]),
                "base_model": identity["model"],
                "target_mode": identity["target_mode"],
                "dataset_id": f"{identity['dataset']}/{identity['frequency']}",
                "horizon": identity["term"],
                "MASE": mase,
                "inference_seconds": inference_seconds,
                "manifest_path": str(run_dir / "manifest.json"),
                "scientific_config": selection.get(
                    "scientific_config",
                    {
                        "model_config": manifest.get("model_config", {}),
                        "pipeline_config": manifest.get("pipeline_config", {}),
                        "experiment_config": manifest.get("experiment_config", {}),
                    },
                ),
            }
        )
    return cells


def summarize_cells(cells: list[dict]) -> list[dict]:
    """Macro-average H settings within datasets, then datasets within models."""
    by_exact_config = defaultdict(list)
    for cell in cells:
        key = (
            cell["model"],
            cell.get("base_model", cell["model"]),
            cell["target_mode"],
            cell["dataset_id"],
            cell["horizon"],
            json.dumps(cell.get("scientific_config", {}), sort_keys=True),
        )
        by_exact_config[key].append(cell)

    config_means = []
    for key, repeats in by_exact_config.items():
        timed = [
            cell["inference_seconds"]
            for cell in repeats
            if cell["inference_seconds"] is not None
            and np.isfinite(cell["inference_seconds"])
        ]
        config_means.append(
            {
                "model": key[0],
                "base_model": key[1],
                "target_mode": key[2],
                "dataset_id": key[3],
                "horizon": key[4],
                "MASE": float(np.mean([cell["MASE"] for cell in repeats])),
                "inference_seconds": (
                    float(np.mean(timed)) if len(timed) == len(repeats) else None
                ),
            }
        )

    by_task = defaultdict(list)
    for cell in config_means:
        by_task[
            (
                cell["model"],
                cell["base_model"],
                cell["target_mode"],
                cell["dataset_id"],
                cell["horizon"],
            )
        ].append(cell)
    effective_cells = []
    for key, configs in by_task.items():
        timed = [cell["inference_seconds"] for cell in configs if cell["inference_seconds"] is not None]
        effective_cells.append(
            {
                "model": key[0],
                "base_model": key[1],
                "target_mode": key[2],
                "dataset_id": key[3],
                "horizon": key[4],
                "MASE": float(np.mean([cell["MASE"] for cell in configs])),
                "inference_seconds": (
                    float(np.mean(timed)) if len(timed) == len(configs) else None
                ),
            }
        )

    by_model = defaultdict(list)
    for cell in effective_cells:
        by_model[cell["model"]].append(cell)

    rows = []
    for model, model_cells in by_model.items():
        mase_by_dataset = defaultdict(list)
        for cell in model_cells:
            mase_by_dataset[cell["dataset_id"]].append(cell["MASE"])

        dataset_mase = [
            float(np.nanmean(values)) for values in mase_by_dataset.values()
        ]
        timed = [
            cell["inference_seconds"]
            for cell in model_cells
            if cell["inference_seconds"] is not None
            and np.isfinite(cell["inference_seconds"])
        ]
        all_tasks_timed = len(timed) == len(model_cells)
        rows.append(
            {
                "model": model,
                "base_model": model_cells[0].get("base_model", model),
                "target_modes": ",".join(
                    sorted({cell["target_mode"] for cell in model_cells})
                ),
                "MASE_macro": float(np.nanmean(dataset_mase)),
                "inference_seconds": float(sum(timed)) if all_tasks_timed else None,
                "datasets": len(mase_by_dataset),
                "tasks": len(model_cells),
                "timed_tasks": len(timed),
            }
        )

    return sorted(rows, key=lambda row: (row["MASE_macro"], row["model"]))


def load_model_statuses(status_dir: Path | None) -> dict[str, dict[str, str]]:
    """Load terminal per-model workflow status for one launch when available."""
    if status_dir is None or not status_dir.is_dir():
        return {}

    statuses = {}
    for status_path in sorted(status_dir.glob("*.status")):
        values = {}
        for line in status_path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key] = value
        statuses[status_path.stem] = values
    return statuses


def parse_model_statuses(values: list[str]) -> dict[str, dict[str, str]]:
    """Parse explicit MODEL=STATE,EXIT_CODE evaluation statuses."""
    statuses = {}
    for value in values:
        model, separator, status = value.partition("=")
        state, comma, exit_code = status.partition(",")
        if not separator or not model or not comma or not state or not exit_code:
            raise ValueError(
                f"Invalid model status {value!r}; expected MODEL=STATE,EXIT_CODE"
            )
        statuses[model] = {"state": state, "exit_code": exit_code}
    return statuses


def add_model_status(
    metric_rows: list[dict],
    models: list[str] | tuple[str, ...],
    statuses: dict[str, dict[str, str]],
    launch_id: str | None,
) -> list[dict]:
    """Attach launch status and retain failed models with no metric cells."""
    rows = [row.copy() for row in metric_rows]
    if statuses:
        present = {row.get("base_model", row["model"]) for row in rows}
        for model in models:
            if model in present:
                continue
            rows.append(
                {
                    "model": model,
                    "base_model": model,
                    "target_modes": "",
                    "MASE_macro": None,
                    "inference_seconds": None,
                    "datasets": 0,
                    "tasks": 0,
                    "timed_tasks": 0,
                }
            )
    for row in rows:
        model = row.get("base_model", row["model"])
        status = statuses.get(model, {})
        row["launch_id"] = launch_id or status.get("launch_id", "")
        row["state"] = status.get("state", "")
        row["exit_code"] = status.get("exit_code", "")
    return sorted(
        rows,
        key=lambda row: (
            row["MASE_macro"] is None,
            float("inf") if row["MASE_macro"] is None else row["MASE_macro"],
            row["model"],
            row["target_modes"],
        ),
    )


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "launch_id",
        "model",
        "base_model",
        "target_modes",
        "state",
        "exit_code",
        "MASE_macro",
        "inference_seconds",
        "datasets",
        "tasks",
        "timed_tasks",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Foundation-model benchmark summary",
        "",
        "MASE is averaged equally over available H settings within each "
        "dataset/frequency, then equally over dataset/frequency entries. "
        "Inference seconds are summed over the same test forecasting tasks; "
        "a blank total means at least one task lacks timing metadata.",
        "",
        "| Model | Target mode | State | Exit | MASE (macro) | Inference seconds | Datasets | Tasks | Timed tasks |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        seconds = row["inference_seconds"]
        mase = row["MASE_macro"]
        lines.append(
            f"| {row['model']} | {row['target_modes']} | {row['state']} | {row['exit_code']} | "
            f"{'' if mase is None else f'{mase:.6f}'} | "
            f"{'' if seconds is None else f'{seconds:.3f}'} | "
            f"{row['datasets']} | {row['tasks']} | {row['timed_tasks']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_manifest(
    cells: list[dict],
    path: Path,
    *,
    results_dir: Path,
    models: list[str],
    target_modes: set[str] | None,
    launch_id: str | None,
    config_filters: dict,
    config_policy: str,
    repeat_policy: str,
    artifacts: list[Path],
) -> None:
    """Record the exact run manifests selected for this aggregate."""
    payload = {
        "schema_version": 1,
        "report": "foundation_model_summary",
        "results_dir": str(results_dir),
        "selection": {
            "models": models,
            "target_modes": sorted(target_modes or []),
            "launch_id": launch_id,
            "config_filters": config_filters,
            "config_policy": config_policy,
            "repeat_policy": repeat_policy,
        },
        "input_manifests": [
            Path(cell["manifest_path"]).resolve().relative_to(results_dir.resolve()).as_posix()
            for cell in cells
        ],
        "artifacts": [str(artifact) for artifact in artifacts],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize foundation-model MASE and test inference time."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=foundation_experiment_root(),
        help="One experiment task root (default: outputs/foundation_models/tasks)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV table (default: <results-dir>/foundation_model_summary.csv)",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Markdown table (default: <results-dir>/foundation_model_summary.md)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Canonical model result directories to summarize",
    )
    parser.add_argument(
        "--launch-id",
        default=None,
        help="Include only task artifacts stamped with this launch ID",
    )
    parser.add_argument(
        "--target-mode",
        nargs="+",
        choices=("univariate", "multivariate"),
        default=None,
        help="Optional target-representation filter",
    )
    parser.add_argument(
        "--run-config",
        action="append",
        default=[],
        help="Manifest filter FIELD=JSON, for example model_config.context_length=2048",
    )
    parser.add_argument(
        "--config-policy",
        choices=("error", "distinct", "latest", "average"),
        default="error",
        help="How to handle different matching scientific configs",
    )
    parser.add_argument(
        "--repeat-policy",
        choices=("selected", "latest", "distinct", "average"),
        default="selected",
        help="How to select or aggregate exact repeated configurations",
    )
    parser.add_argument(
        "--status-dir",
        type=Path,
        default=None,
        help="Per-model workflow status directory for the selected launch",
    )
    parser.add_argument(
        "--model-status",
        action="append",
        default=[],
        metavar="MODEL=STATE,EXIT_CODE",
        help=(
            "Explicit evaluation status, used when the workflow status filename "
            "is a comparison mode rather than a model alias"
        ),
    )
    args = parser.parse_args()

    args.csv = args.csv or args.results_dir / "foundation_model_summary.csv"
    args.markdown = args.markdown or args.results_dir / "foundation_model_summary.md"
    models = set(args.models)
    config_filters = parse_config_filters(args.run_config)
    cells = load_result_cells(
        args.results_dir,
        models,
        launch_id=args.launch_id,
        target_modes=None if args.target_mode is None else set(args.target_mode),
        config_filters=config_filters,
        config_policy=args.config_policy,
        repeat_policy=args.repeat_policy,
    )
    metric_rows = summarize_cells(cells)
    statuses = load_model_statuses(args.status_dir)
    statuses.update(parse_model_statuses(args.model_status))
    rows = add_model_status(metric_rows, args.models, statuses, args.launch_id)
    if not rows:
        raise SystemExit(
            f"No aggregate metric cells or model statuses found below {args.results_dir}"
        )

    write_csv(rows, args.csv)
    write_markdown(rows, args.markdown)
    write_report_manifest(
        cells,
        args.csv.with_name("foundation_model_report_manifest.json"),
        results_dir=args.results_dir,
        models=args.models,
        target_modes=None if args.target_mode is None else set(args.target_mode),
        launch_id=args.launch_id,
        config_filters=config_filters,
        config_policy=args.config_policy,
        repeat_policy=args.repeat_policy,
        artifacts=[args.csv, args.markdown],
    )
    print(f"Foundation-model summary written to {args.csv} and {args.markdown}")
    print()
    for row in rows:
        seconds = row["inference_seconds"]
        seconds_text = "incomplete" if seconds is None else f"{seconds:.3f}s"
        mase = row["MASE_macro"]
        mase_text = "incomplete" if mase is None else f"{mase:.6f}"
        print(
            f"{(row['model'] + ('/' + row['target_modes'] if row['target_modes'] else '')):<28} "
            f"state={row['state'] or 'unknown'}  "
            f"MASE={mase_text}  "
            f"inference={seconds_text}  "
            f"coverage={row['timed_tasks']}/{row['tasks']} timed tasks"
        )


if __name__ == "__main__":
    main()
