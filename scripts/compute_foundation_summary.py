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

DEFAULT_MODELS = (
    "chronos_bolt",
    "chronos2",
    "tirex",
    "toto",
    "ts_icl",
    "seasonal_naive",
)


def load_result_cells(root: Path, models: set[str] | None = None) -> list[dict]:
    """Load one dataset/frequency/horizon cell from every complete result directory."""
    cells = []
    for metrics_path in sorted(root.glob("*/*/*/*/metrics.npz")):
        relative = metrics_path.relative_to(root)
        if len(relative.parts) != 5:
            continue
        model, dataset, freq, horizon, _ = relative.parts
        if models is not None and model not in models:
            continue

        with np.load(metrics_path) as metrics:
            if "MASE" not in metrics.files:
                continue
            mase = float(np.nanmean(metrics["MASE"]))

        config_path = metrics_path.with_name("config.json")
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else {}
        )
        inference_seconds = config.get("inference_seconds")
        if inference_seconds is not None:
            inference_seconds = float(inference_seconds)

        cells.append(
            {
                "model": model,
                "dataset_id": f"{dataset}/{freq}",
                "horizon": horizon,
                "MASE": mase,
                "inference_seconds": inference_seconds,
            }
        )
    return cells


def summarize_cells(cells: list[dict]) -> list[dict]:
    """Macro-average H settings within datasets, then datasets within models."""
    by_model = defaultdict(list)
    for cell in cells:
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
                "MASE_macro": float(np.nanmean(dataset_mase)),
                "inference_seconds": float(sum(timed)) if all_tasks_timed else None,
                "datasets": len(mase_by_dataset),
                "tasks": len(model_cells),
                "timed_tasks": len(timed),
            }
        )

    return sorted(rows, key=lambda row: (row["MASE_macro"], row["model"]))


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
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
        "| Model | MASE (macro) | Inference seconds | Datasets | Tasks | Timed tasks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        seconds = row["inference_seconds"]
        lines.append(
            f"| {row['model']} | {row['MASE_macro']:.6f} | "
            f"{'' if seconds is None else f'{seconds:.3f}'} | "
            f"{row['datasets']} | {row['tasks']} | {row['timed_tasks']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from timebench.paths import outputs_root, results_root

    parser = argparse.ArgumentParser(
        description="Summarize foundation-model MASE and test inference time."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=results_root(),
        help="TIME result root (default: TIME_OUTPUTS/results)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=outputs_root() / "foundation_model_summary.csv",
        help="CSV table to write",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=outputs_root() / "foundation_model_summary.md",
        help="Markdown table to write",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Canonical model result directories to summarize",
    )
    args = parser.parse_args()

    models = set(args.models)
    rows = summarize_cells(load_result_cells(args.results_dir, models))
    if not rows:
        raise SystemExit(f"No MASE result cells found below {args.results_dir}")

    write_csv(rows, args.csv)
    write_markdown(rows, args.markdown)
    print(f"Foundation-model summary written to {args.csv} and {args.markdown}")
    print()
    for row in rows:
        seconds = row["inference_seconds"]
        seconds_text = "incomplete" if seconds is None else f"{seconds:.3f}s"
        print(
            f"{row['model']:<28} MASE={row['MASE_macro']:.6f}  "
            f"inference={seconds_text}  "
            f"coverage={row['timed_tasks']}/{row['tasks']} timed tasks"
        )


if __name__ == "__main__":
    main()
