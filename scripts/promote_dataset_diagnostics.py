#!/usr/bin/env python
"""Promote one pre-metadata diagnostics run into the shared compact layout."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


COUNT_FIELDS = (
    "generated_queries",
    "channel_windows",
    "values",
    "finite_values",
    "nan_values",
    "positive_infinity_values",
    "negative_infinity_values",
    "nonfinite_values",
    "nonfinite_channel_windows",
    "constant_channel_windows",
    "all_nonfinite_channel_windows",
)
SUMMARY_FIELDS = (
    "dataset",
    "frequency",
    "term",
    "scope",
    "window_config",
    "context_limit",
    "prediction_length",
    *COUNT_FIELDS,
)
EVENT_FIELDS = (
    "dataset",
    "frequency",
    "term",
    "item_id",
    "window_index",
    "scope",
    "window_config",
    "context_limit",
    "raw_context_length",
    "window_length",
    "prediction_length",
    "field",
    "channel_index",
    "channel_name",
    "source_start_position",
    "source_end_position",
    "source_start_timestamp",
    "source_end_timestamp",
    "finite_values",
    "nan_values",
    "positive_infinity_values",
    "negative_infinity_values",
    "nonfinite_values",
    "constant",
    "all_nonfinite",
    "constant_value",
)
POSITION_FIELDS = (
    "dataset",
    "frequency",
    "item_id",
    "field",
    "channel_index",
    "channel_name",
    "source_position",
    "timestamp",
    "value_kind",
)
MODEL_CONTEXT_FIELDS = ("profile_name", "context_limit", "window_config_prefix")


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as input_file:
        yield from csv.DictReader(input_file)


def write_rows(path: Path, fieldnames: tuple[str, ...], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def window_config(row: dict[str, str]) -> str:
    horizon = row["prediction_length"]
    if row["scope"] == "query":
        return f"H={horizon}"
    limit = row.get("context_limit", "") or "full"
    return f"L={limit},H={horizon}"


def normalize_task_rows(source: Path) -> list[dict[str, str | int]]:
    normalized: dict[tuple, dict[str, str | int]] = {}
    for row in read_rows(source):
        output = {
            "dataset": row["dataset"],
            "frequency": row["frequency"],
            "term": row["term"],
            "scope": row["scope"],
            "window_config": window_config(row),
            "context_limit": (
                "" if row["scope"] == "query" else row.get("context_limit", "")
            ),
            "prediction_length": row["prediction_length"],
            **{field: int(row[field]) for field in COUNT_FIELDS},
        }
        key = tuple(output[field] for field in SUMMARY_FIELDS[:7])
        previous = normalized.get(key)
        if previous is not None and previous != output:
            raise ValueError(f"Conflicting duplicate L-H summary rows: {key}")
        normalized[key] = output
    return [normalized[key] for key in sorted(normalized)]


def aggregate_dataset_rows(task_rows: list[dict]) -> list[dict]:
    counts = defaultdict(lambda: {field: 0 for field in COUNT_FIELDS})
    metadata = {}
    for row in task_rows:
        key = (
            row["dataset"],
            row["frequency"],
            row["scope"],
            row["window_config"],
            row["context_limit"],
            row["prediction_length"],
        )
        metadata[key] = {
            "dataset": row["dataset"],
            "frequency": row["frequency"],
            "term": "all",
            "scope": row["scope"],
            "window_config": row["window_config"],
            "context_limit": row["context_limit"],
            "prediction_length": row["prediction_length"],
        }
        for field in COUNT_FIELDS:
            counts[key][field] += int(row[field])
    return [{**metadata[key], **counts[key]} for key in sorted(counts)]


def normalize_events(source: Path, destination: Path) -> None:
    seen = set()
    with destination.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        for row in read_rows(source):
            output = {field: row.get(field, "") for field in EVENT_FIELDS}
            output["window_config"] = window_config(row)
            if output["scope"] == "query":
                output["context_limit"] = ""
            key = tuple(output[field] for field in EVENT_FIELDS)
            if key not in seen:
                seen.add(key)
                writer.writerow(output)


def normalize_positions(source: Path, destination: Path) -> None:
    seen = set()
    with destination.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=POSITION_FIELDS)
        writer.writeheader()
        for row in read_rows(source):
            output = {field: row.get(field, "") for field in POSITION_FIELDS}
            key = (
                output["dataset"],
                output["frequency"],
                output["item_id"],
                output["field"],
                output["channel_index"],
                output["source_position"],
                output["value_kind"],
            )
            if key not in seen:
                seen.add(key)
                writer.writerow(output)


def model_context_rows(profiles: dict[str, int | None]) -> list[dict]:
    return [
        {
            "profile_name": name,
            "context_limit": "" if limit is None else limit,
            "window_config_prefix": "L=full" if limit is None else f"L={limit}",
        }
        for name, limit in profiles.items()
    ]


def copy_full_features(source: Path, destination: Path) -> None:
    selected_names = {"full.csv", "full_dataset.csv", "dataset_features_full.csv"}
    for artifact in source.rglob("*"):
        if not artifact.is_file() or artifact.name not in selected_names:
            continue
        target = destination / artifact.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--log-export-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    source_manifest_path = args.source_audit / "audit_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "completed":
        raise ValueError(f"Audit is not complete: {source_manifest_path}")
    feature_index = args.source_features / "dataset_features_full.csv"
    if not feature_index.is_file():
        raise FileNotFoundError(f"Completed feature index not found: {feature_index}")

    audit_root = args.metadata_root / "window_audit"
    feature_root = args.metadata_root / "stl_features"
    audit_root.mkdir(parents=True, exist_ok=True)
    args.log_export_dir.mkdir(parents=True, exist_ok=True)

    profiles = dict(source_manifest["context_profiles"])
    if None in profiles.values():
        profiles.setdefault("seasonal_naive", None)

    task_rows = normalize_task_rows(args.source_audit / "task_summary.csv")
    dataset_rows = aggregate_dataset_rows(task_rows)
    write_rows(audit_root / "task_summary.csv", SUMMARY_FIELDS, task_rows)
    write_rows(audit_root / "dataset_summary.csv", SUMMARY_FIELDS, dataset_rows)
    write_rows(
        audit_root / "model_contexts.csv",
        MODEL_CONTEXT_FIELDS,
        model_context_rows(profiles),
    )
    normalize_events(args.source_audit / "window_events.csv", audit_root / "window_events.csv")
    normalize_positions(
        args.source_audit / "nonfinite_positions.csv",
        audit_root / "nonfinite_positions.csv",
    )
    copy_full_features(args.source_features, feature_root)

    totals = {field: sum(int(row[field]) for row in task_rows) for field in COUNT_FIELDS}
    promoted_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        **source_manifest,
        "schema_version": 1,
        "storage": "shared_dataset_metadata",
        "promoted_from": str(args.source_audit.resolve()),
        "promoted_at": promoted_at,
        "context_profiles": profiles,
        "distinct_context_limits": list(dict.fromkeys(profiles.values())),
        "artifacts": [
            "window_events.csv",
            "nonfinite_positions.csv",
            "task_summary.csv",
            "dataset_summary.csv",
            "model_contexts.csv",
        ],
        "window_configuration_totals": totals,
    }
    manifest.pop("profile_expanded_totals", None)
    (audit_root / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    aggregate_files = (
        audit_root / "audit_manifest.json",
        audit_root / "model_contexts.csv",
        audit_root / "task_summary.csv",
        audit_root / "dataset_summary.csv",
        feature_root / "dataset_features_full.csv",
    )
    for artifact in aggregate_files:
        shutil.copy2(artifact, args.log_export_dir / artifact.name)
    export_manifest = {
        "schema_version": 1,
        "report": "dataset_metadata_export",
        "job_id": args.job_id,
        "launch_id": source_manifest.get("launch_id"),
        "created_at": promoted_at,
        "metadata_root": str(args.metadata_root.resolve()),
        "aggregates": [artifact.name for artifact in aggregate_files],
    }
    (args.log_export_dir / "metadata_export_manifest.json").write_text(
        json.dumps(export_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Promoted shared metadata to {args.metadata_root}")
    print(f"Exported compact aggregates to {args.log_export_dir}")


if __name__ == "__main__":
    main()
