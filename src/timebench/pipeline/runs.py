"""Plain-config manifests for TIME experiment runs."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from timebench.paths import PROJECT_ROOT


SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
VALID_STATUSES = {"running", "interrupted", "completed"}
CONFIG_POLICIES = ("error", "latest")
RUN_PATTERN = re.compile(r"run_(\d+)")


class ManifestError(ValueError):
    """Raised when run identity or selection is ambiguous or invalid."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_index(run_dir: Path) -> int:
    match = RUN_PATTERN.fullmatch(run_dir.name)
    if match is None:
        raise ManifestError(f"Run directory must be named run_<n>: {run_dir}")
    return int(match.group(1))


def load_manifest(path_or_run: str | Path) -> dict[str, Any]:
    """Load and validate one current manifest."""
    path = Path(path_or_run)
    if path.is_dir():
        path = path / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(
            f"{path} uses schema_version={manifest.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if manifest.get("status") not in VALID_STATUSES:
        raise ManifestError(f"{path} has invalid status {manifest.get('status')!r}")
    _run_index(path.parent)
    return manifest


@dataclass
class RunHandle:
    """One allocated run that becomes completed only after artifact validation."""

    run_dir: Path
    manifest: dict[str, Any]
    _completed: bool = False

    def __enter__(self) -> "RunHandle":
        return self

    def complete(self, required_artifacts: Sequence[str]) -> None:
        artifacts = [str(value) for value in required_artifacts]
        if not artifacts:
            raise ManifestError(f"Completed run has no required artifacts: {self.run_dir}")
        missing = [
            name
            for name in artifacts
            if not (self.run_dir / name).is_file()
            or (self.run_dir / name).stat().st_size == 0
        ]
        if missing:
            raise ManifestError(
                f"Run cannot be completed with missing or empty artifacts: {missing}"
            )
        self.manifest["required_artifacts"] = artifacts
        self.manifest["status"] = "completed"
        self.manifest["completed_at"] = _now()
        self.manifest["updated_at"] = self.manifest["completed_at"]
        _write_manifest(self.run_dir / MANIFEST_NAME, self.manifest)
        self._completed = True

    def __exit__(self, error_type, error, traceback) -> bool:
        if error_type is not None or not self._completed:
            self.manifest["status"] = "interrupted"
            self.manifest["updated_at"] = _now()
            if error_type is not None:
                self.manifest["error"] = {
                    "type": error_type.__name__,
                    "message": str(error),
                }
            elif "error" not in self.manifest:
                self.manifest["error"] = {
                    "type": "IncompleteRun",
                    "message": "run exited without declaring complete artifacts",
                }
            _write_manifest(self.run_dir / MANIFEST_NAME, self.manifest)
        return False


def allocate_run(
    identity_root: str | Path,
    *,
    experiment: str,
    identity: Mapping[str, Any],
    model_config: Mapping[str, Any],
    pipeline_config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> RunHandle:
    """Allocate the next ``run_n`` and write its plain configuration."""
    root = Path(identity_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    indices = [
        _run_index(path)
        for path in root.iterdir()
        if path.is_dir() and RUN_PATTERN.fullmatch(path.name)
    ]
    run_dir = root / f"run_{0 if not indices else max(indices) + 1}"
    run_dir.mkdir()
    started_at = _now()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "project": PROJECT_ROOT.name,
        "experiment": str(experiment),
        "identity": dict(identity),
        "model_config": dict(model_config),
        "pipeline_config": dict(pipeline_config),
        "runtime_config": dict(runtime_config),
        "experiment_config": dict(experiment_config),
        "provenance": dict(provenance or {}),
        "launch": {
            "launch_id": os.environ.get("TIME_LAUNCH_ID"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "status": "running",
        "required_artifacts": [],
        "started_at": started_at,
        "updated_at": started_at,
    }
    _write_manifest(run_dir / MANIFEST_NAME, manifest)
    return RunHandle(run_dir, manifest)


def _nested_value(manifest: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = manifest
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def parse_config_filters(values: Sequence[str] | None) -> dict[str, Any]:
    """Parse repeatable ``manifest.path=JSON`` aggregate filters."""
    filters: dict[str, Any] = {}
    for value in values or []:
        key, separator, raw = value.partition("=")
        if not separator or not key:
            raise ManifestError(f"Run-config filter must be FIELD=JSON, got {value!r}")
        filters[key] = json.loads(raw)
    return filters


def resolve_target_mode(
    requested: str,
    *,
    target_dim: int,
    supports_multivariate: bool,
) -> str:
    """Resolve ``auto`` to the representation actually passed to the model."""
    if requested not in {"auto", "univariate", "multivariate"}:
        raise ValueError(
            "target_mode must be auto, univariate, or multivariate"
        )
    if requested == "multivariate" and not supports_multivariate:
        raise ValueError("This model does not support multivariate target inputs")
    if requested == "multivariate" and target_dim < 2:
        raise ValueError("A one-channel dataset cannot run in multivariate mode")
    if requested == "auto":
        if supports_multivariate and target_dim > 1:
            return "multivariate"
        return "univariate"
    return requested


def _scientific_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_config": manifest.get("model_config", {}),
        "pipeline_config": manifest.get("pipeline_config", {}),
        "experiment_config": manifest.get("experiment_config", {}),
    }


def _same_config_groups(
    manifests: Sequence[tuple[Path, dict[str, Any]]],
) -> list[list[tuple[Path, dict[str, Any]]]]:
    groups: list[list[tuple[Path, dict[str, Any]]]] = []
    for item in manifests:
        config = _scientific_config(item[1])
        for group in groups:
            if _scientific_config(group[0][1]) == config:
                group.append(item)
                break
        else:
            groups.append([item])
    return groups


def select_completed_runs(
    root: str | Path,
    *,
    models: set[str] | None = None,
    target_modes: set[str] | None = None,
    launch_id: str | None = None,
    config_filters: Mapping[str, Any] | None = None,
    config_policy: str = "error",
) -> list[tuple[Path, dict[str, Any]]]:
    """Select one completed run per task identity without mixing configs."""
    if config_policy not in CONFIG_POLICIES:
        raise ManifestError(f"config_policy must be one of {CONFIG_POLICIES}")
    filters = dict(config_filters or {})
    candidates: list[tuple[Path, dict[str, Any]]] = []
    root = Path(root).expanduser().resolve()
    if not root.exists():
        return []
    for path in sorted(root.rglob(MANIFEST_NAME)):
        manifest = load_manifest(path)
        if manifest["status"] != "completed":
            continue
        identity = manifest.get("identity", {})
        if models is not None and identity.get("model") not in models:
            continue
        if target_modes is not None and identity.get("target_mode") not in target_modes:
            continue
        if launch_id is not None and manifest.get("launch", {}).get("launch_id") != launch_id:
            continue
        if any(_nested_value(manifest, key) != value for key, value in filters.items()):
            continue
        candidates.append((path.parent, manifest))

    by_identity: list[list[tuple[Path, dict[str, Any]]]] = []
    for item in candidates:
        identity = item[1].get("identity", {})
        for group in by_identity:
            if group[0][1].get("identity", {}) == identity:
                group.append(item)
                break
        else:
            by_identity.append([item])

    selected: list[tuple[Path, dict[str, Any]]] = []
    for identity_group in by_identity:
        config_groups = _same_config_groups(identity_group)
        if len(config_groups) > 1 and config_policy == "error":
            identity = identity_group[0][1]["identity"]
            raise ManifestError(
                "Multiple scientific run configurations match "
                f"{identity}; use --run-config or --config-policy latest"
            )
        pool = identity_group if config_policy == "latest" else config_groups[0]
        selected.append(max(pool, key=lambda item: _run_index(item[0])))

    experiments = {item[1].get("experiment") for item in selected}
    if len(experiments) > 1:
        raise ManifestError(
            f"Selected runs cross experiment roots: {sorted(experiments)}; "
            "point the result reader at one experiment root"
        )

    by_task: list[list[tuple[Path, dict[str, Any]]]] = []
    for item in selected:
        identity = item[1]["identity"]
        key = (
            identity["model"],
            identity["dataset"],
            identity["frequency"],
            identity["term"],
        )
        for group in by_task:
            sample = group[0][1]["identity"]
            sample_key = (
                sample["model"],
                sample["dataset"],
                sample["frequency"],
                sample["term"],
            )
            if key == sample_key:
                group.append(item)
                break
        else:
            by_task.append([item])
    for group in by_task:
        modes = {item[1]["identity"]["target_mode"] for item in group}
        if len(modes) > 1:
            identity = group[0][1]["identity"]
            raise ManifestError(
                "Multiple target modes match "
                f"{identity['model']} {identity['dataset']}/{identity['frequency']} "
                f"{identity['term']}: {sorted(modes)}; use --target-mode"
            )

    if config_policy == "error":
        by_model: list[list[tuple[Path, dict[str, Any]]]] = []
        for item in selected:
            identity = item[1]["identity"]
            key = identity["model"]
            for group in by_model:
                sample_identity = group[0][1]["identity"]
                if key == sample_identity["model"]:
                    group.append(item)
                    break
            else:
                by_model.append([item])
        for group in by_model:
            global_configs = [
                {
                    "model_config": item[1].get("model_config", {}),
                    "covariate_mode": item[1]
                    .get("experiment_config", {})
                    .get("covariate_mode"),
                }
                for item in group
            ]
            if any(config != global_configs[0] for config in global_configs[1:]):
                identity = group[0][1]["identity"]
                raise ManifestError(
                    "Selected tasks mix model or experiment configurations for "
                    f"{identity['model']}; use "
                    "--run-config to select one configuration"
                )
    return sorted(
        selected,
        key=lambda item: tuple(str(value) for value in item[1]["identity"].values()),
    )
