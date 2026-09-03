"""Audit the exact TIME evaluation queries and model context windows."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from timebench.evaluation.data import (
    DEFAULT_CONFIG_PATH,
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.evaluation.utils import get_available_terms
from timebench.paths import outputs_root


DEFAULT_CONTEXT_PROFILES = {
    "full": None,
    "chronos2": 8192,
    "ts_icl": 4096,
    "chronos_bolt": 2048,
}

SUMMARY_FIELDS = (
    "dataset",
    "frequency",
    "term",
    "scope",
    "context_profile",
    "context_limit",
    "prediction_length",
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

EVENT_FIELDS = (
    "dataset",
    "frequency",
    "term",
    "item_id",
    "window_index",
    "scope",
    "context_profile",
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
    "term",
    "item_id",
    "window_index",
    "scope",
    "context_profile",
    "field",
    "channel_index",
    "channel_name",
    "window_position",
    "source_position",
    "timestamp",
    "value_kind",
)


def _log(message: str) -> None:
    print(f"{datetime.now().astimezone().isoformat(timespec='seconds')} | {message}")


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_context_profiles(values: Iterable[str] | None) -> dict[str, int | None]:
    if not values:
        return dict(DEFAULT_CONTEXT_PROFILES)

    profiles: dict[str, int | None] = {}
    for value in values:
        name, separator, raw_limit = value.partition("=")
        name = name.strip()
        if not name or name in profiles:
            raise ValueError(f"Invalid or duplicate context profile: {value!r}")
        if not separator:
            if name != "full":
                raise ValueError(
                    f"Context profile {value!r} needs NAME=LENGTH; only full has no limit"
                )
            profiles[name] = None
            continue
        limit = int(raw_limit)
        if limit <= 0:
            raise ValueError(f"Context length must be positive: {value!r}")
        profiles[name] = limit
    return profiles


def _channels(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError(f"Expected target shape (channels, time), received {array.shape}")
    return array


def _channel_names(entry: dict, channels: int) -> list[str]:
    names = entry.get("variate_names")
    if names is not None and len(names) == channels:
        return [str(name) for name in names]
    if channels == 1:
        return ["target"]
    return [f"dim_{index}" for index in range(channels)]


def _timestamp(start, frequency: str, source_position: int) -> str:
    try:
        if isinstance(start, pd.Period):
            return str(start + source_position)
        timestamp = pd.Timestamp(start)
        return str(timestamp + source_position * pd.tseries.frequencies.to_offset(frequency))
    except (TypeError, ValueError, OverflowError):
        return ""


def _empty_counts() -> dict[str, int]:
    return {
        "generated_queries": 0,
        "channel_windows": 0,
        "values": 0,
        "finite_values": 0,
        "nan_values": 0,
        "positive_infinity_values": 0,
        "negative_infinity_values": 0,
        "nonfinite_values": 0,
        "nonfinite_channel_windows": 0,
        "constant_channel_windows": 0,
        "all_nonfinite_channel_windows": 0,
    }


def _add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for field in target:
        target[field] += source[field]


def _inspect_channel(values: np.ndarray) -> tuple[dict[str, int], np.ndarray, bool, bool, float | None]:
    nan_mask = np.isnan(values)
    positive_infinity_mask = np.isposinf(values)
    negative_infinity_mask = np.isneginf(values)
    nonfinite_mask = ~np.isfinite(values)
    finite_values = values[~nonfinite_mask]
    constant = bool(
        values.size > 0
        and finite_values.size == values.size
        and np.all(finite_values == finite_values[0])
    )
    all_nonfinite = bool(values.size > 0 and finite_values.size == 0)
    counts = _empty_counts()
    counts.update(
        {
            "channel_windows": 1,
            "values": int(values.size),
            "finite_values": int(finite_values.size),
            "nan_values": int(nan_mask.sum()),
            "positive_infinity_values": int(positive_infinity_mask.sum()),
            "negative_infinity_values": int(negative_infinity_mask.sum()),
            "nonfinite_values": int(nonfinite_mask.sum()),
            "nonfinite_channel_windows": int(nonfinite_mask.any()),
            "constant_channel_windows": int(constant),
            "all_nonfinite_channel_windows": int(all_nonfinite),
        }
    )
    constant_value = float(finite_values[0]) if constant else None
    return counts, nonfinite_mask, constant, all_nonfinite, constant_value


def audit_time_windows(
    *,
    config_path: Path | None,
    output_dir: Path,
    context_profiles: dict[str, int | None],
    selected_datasets: set[str] | None = None,
) -> dict:
    """Audit targets from the same GluonTS queries consumed by TIME runners."""
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "window_events.csv"
    positions_path = output_dir / "nonfinite_positions.csv"
    task_summary_path = output_dir / "task_summary.csv"
    dataset_summary_path = output_dir / "dataset_summary.csv"
    manifest_path = output_dir / "audit_manifest.json"

    resolved_config_path = config_path or DEFAULT_CONFIG_PATH
    config = load_dataset_config(resolved_config_path)
    datasets_config = config.get("datasets", {})
    dataset_keys = [
        key for key in datasets_config if selected_datasets is None or key in selected_datasets
    ]
    if selected_datasets is not None:
        missing = sorted(selected_datasets.difference(dataset_keys))
        if missing:
            raise ValueError(f"Datasets are absent from the TIME config: {missing}")

    launch_id = os.environ.get("TIME_LAUNCH_ID")
    started_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "report": "time_window_audit",
        "status": "running",
        "launch_id": launch_id,
        "started_at": started_at,
        "updated_at": started_at,
        "config_path": str(resolved_config_path.resolve()),
        "datasets": dataset_keys,
        "context_profiles": context_profiles,
        "definitions": {
            "query": "The exact GluonTS label target for one configured test instance.",
            "context": "The exact input target, truncated from the left to the profile limit.",
            "constant": "A non-empty channel window whose values are all finite and exactly equal.",
            "position": "Zero-based position in the original saved-Arrow target series.",
        },
        "artifacts": [
            events_path.name,
            positions_path.name,
            task_summary_path.name,
            dataset_summary_path.name,
        ],
    }
    _write_json(manifest_path, manifest)

    summaries: dict[tuple, dict[str, int]] = defaultdict(_empty_counts)
    summary_metadata: dict[tuple, dict] = {}

    try:
        with events_path.open("w", newline="", encoding="utf-8") as events_file, positions_path.open(
            "w", newline="", encoding="utf-8"
        ) as positions_file:
            event_writer = csv.DictWriter(events_file, fieldnames=EVENT_FIELDS)
            position_writer = csv.DictWriter(positions_file, fieldnames=POSITION_FIELDS)
            event_writer.writeheader()
            position_writer.writeheader()

            for dataset_number, dataset_key in enumerate(dataset_keys, start=1):
                terms = get_available_terms(dataset_key, config)
                _log(
                    f"dataset {dataset_number}/{len(dataset_keys)} {dataset_key} terms={terms}"
                )
                for term in terms:
                    settings = get_dataset_settings(dataset_key, term, config)
                    prediction_length = int(settings["prediction_length"])
                    dataset = Dataset(
                        name=dataset_key,
                        term=term,
                        to_univariate=False,
                        prediction_length=prediction_length,
                        test_length=int(settings["test_length"]),
                        val_length=int(settings["val_length"]),
                    )
                    item_window_indices: dict[str, int] = defaultdict(int)

                    for input_entry, label_entry in dataset.test_data:
                        item_id = str(input_entry.get("item_id", "unknown"))
                        window_index = item_window_indices[item_id]
                        item_window_indices[item_id] += 1
                        context = _channels(input_entry["target"])
                        query = _channels(label_entry["target"])
                        if context.shape[0] != query.shape[0]:
                            raise ValueError(
                                f"Target channel count changes at the query boundary for "
                                f"{dataset_key}/{term}/{item_id}"
                            )
                        names = _channel_names(input_entry, context.shape[0])
                        raw_context_length = int(context.shape[-1])
                        start = input_entry.get("start")

                        scopes = [
                            (
                                "query",
                                "forecast_horizon",
                                prediction_length,
                                query,
                                raw_context_length,
                            )
                        ]
                        for profile, limit in context_profiles.items():
                            source_start = (
                                0 if limit is None else max(0, raw_context_length - limit)
                            )
                            scopes.append(
                                (
                                    "context",
                                    profile,
                                    limit,
                                    context[:, source_start:],
                                    source_start,
                                )
                            )

                        for scope, profile, limit, block, source_start in scopes:
                            key = (dataset_key, term, scope, profile)
                            if key not in summary_metadata:
                                summary_metadata[key] = {
                                    "dataset": dataset_key.rpartition("/")[0],
                                    "frequency": dataset.freq,
                                    "term": term,
                                    "scope": scope,
                                    "context_profile": profile,
                                    "context_limit": "" if limit is None else limit,
                                    "prediction_length": prediction_length,
                                }
                            summaries[key]["generated_queries"] += 1

                            for channel_index, values in enumerate(block):
                                counts, nonfinite_mask, constant, all_nonfinite, constant_value = (
                                    _inspect_channel(values)
                                )
                                _add_counts(summaries[key], counts)
                                if not nonfinite_mask.any() and not constant:
                                    continue

                                source_end = source_start + len(values) - 1
                                event_writer.writerow(
                                    {
                                        **summary_metadata[key],
                                        "item_id": item_id,
                                        "window_index": window_index,
                                        "raw_context_length": raw_context_length,
                                        "window_length": len(values),
                                        "field": "target",
                                        "channel_index": channel_index,
                                        "channel_name": names[channel_index],
                                        "source_start_position": source_start,
                                        "source_end_position": source_end,
                                        "source_start_timestamp": _timestamp(
                                            start, dataset.freq, source_start
                                        ),
                                        "source_end_timestamp": _timestamp(
                                            start, dataset.freq, source_end
                                        ),
                                        "finite_values": counts["finite_values"],
                                        "nan_values": counts["nan_values"],
                                        "positive_infinity_values": counts[
                                            "positive_infinity_values"
                                        ],
                                        "negative_infinity_values": counts[
                                            "negative_infinity_values"
                                        ],
                                        "nonfinite_values": counts["nonfinite_values"],
                                        "constant": str(constant).lower(),
                                        "all_nonfinite": str(all_nonfinite).lower(),
                                        "constant_value": (
                                            "" if constant_value is None else constant_value
                                        ),
                                    }
                                )

                                for window_position in np.flatnonzero(nonfinite_mask):
                                    value = values[window_position]
                                    if np.isnan(value):
                                        value_kind = "nan"
                                    elif np.isposinf(value):
                                        value_kind = "positive_infinity"
                                    else:
                                        value_kind = "negative_infinity"
                                    source_position = source_start + int(window_position)
                                    position_writer.writerow(
                                        {
                                            "dataset": dataset_key.rpartition("/")[0],
                                            "frequency": dataset.freq,
                                            "term": term,
                                            "item_id": item_id,
                                            "window_index": window_index,
                                            "scope": scope,
                                            "context_profile": profile,
                                            "field": "target",
                                            "channel_index": channel_index,
                                            "channel_name": names[channel_index],
                                            "window_position": int(window_position),
                                            "source_position": source_position,
                                            "timestamp": _timestamp(
                                                start, dataset.freq, source_position
                                            ),
                                            "value_kind": value_kind,
                                        }
                                    )

                    _log(
                        f"task {dataset_key}/{term} audited queries="
                        f"{sum(item_window_indices.values())}"
                    )

        task_rows = []
        for key in sorted(summaries):
            task_rows.append({**summary_metadata[key], **summaries[key]})
        with task_summary_path.open("w", newline="", encoding="utf-8") as summary_file:
            writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(task_rows)

        dataset_counts: dict[tuple, dict[str, int]] = defaultdict(_empty_counts)
        dataset_metadata: dict[tuple, dict] = {}
        for row in task_rows:
            key = (
                row["dataset"],
                row["frequency"],
                row["scope"],
                row["context_profile"],
            )
            dataset_metadata[key] = {
                "dataset": row["dataset"],
                "frequency": row["frequency"],
                "term": "all",
                "scope": row["scope"],
                "context_profile": row["context_profile"],
                "context_limit": row["context_limit"],
                "prediction_length": "",
            }
            _add_counts(dataset_counts[key], row)
        dataset_rows = [
            {**dataset_metadata[key], **dataset_counts[key]}
            for key in sorted(dataset_counts)
        ]
        with dataset_summary_path.open("w", newline="", encoding="utf-8") as summary_file:
            writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(dataset_rows)

        totals = _empty_counts()
        for row in task_rows:
            _add_counts(totals, row)
        completed_at = datetime.now(timezone.utc).isoformat()
        manifest.update(
            {
                "status": "completed",
                "updated_at": completed_at,
                "completed_at": completed_at,
                "task_summary_rows": len(task_rows),
                "dataset_summary_rows": len(dataset_rows),
                "profile_expanded_totals": totals,
            }
        )
        _write_json(manifest_path, manifest)
        _log(
            f"audit completed output={output_dir} nonfinite_values="
            f"{totals['nonfinite_values']} constant_channel_windows="
            f"{totals['constant_channel_windows']}"
        )
        return manifest
    except Exception as error:
        failed_at = datetime.now(timezone.utc).isoformat()
        manifest.update(
            {
                "status": "interrupted",
                "updated_at": failed_at,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        _write_json(manifest_path, manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit TIME test queries and model-effective target contexts."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=outputs_root() / "dataset_diagnostics" / "manual",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Audit one configured dataset/frequency key; repeat to select several.",
    )
    parser.add_argument(
        "--context-profile",
        action="append",
        default=None,
        help="Context profile NAME=LENGTH; use full without a length.",
    )
    args = parser.parse_args()
    audit_time_windows(
        config_path=args.config,
        output_dir=args.output_dir,
        context_profiles=_parse_context_profiles(args.context_profile),
        selected_datasets=None if args.dataset is None else set(args.dataset),
    )


if __name__ == "__main__":
    main()
