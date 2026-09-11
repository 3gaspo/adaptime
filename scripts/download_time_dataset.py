#!/usr/bin/env python3
"""Download the official TIME Arrow dataset for offline cluster evaluation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# DGX cannot use the Hub's Xet/CAS transfer path. This must be set before the
# huggingface_hub import because Hub environment variables are read at import.
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import HfApi, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from timebench.paths import dataset_storage_root


DEFAULT_REPO_ID = "Real-TSF/TIME"
REVISION_FILE = ".time_snapshot_revision"


def download_time_dataset(
    destination: Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = "main",
    resume: bool = False,
    max_workers: int = 4,
) -> tuple[str, int]:
    """Download one immutable repository revision and validate saved Arrow datasets."""
    destination = destination.expanduser().resolve()
    revision_path = destination / REVISION_FILE
    existing_revision = (
        revision_path.read_text(encoding="utf-8").strip()
        if revision_path.is_file()
        else None
    )
    destination_has_files = destination.exists() and any(destination.iterdir())
    if destination_has_files and not resume:
        raise FileExistsError(
            f"TIME destination is not empty: {destination}. "
            "Pass --resume only for an interrupted download of this snapshot."
        )

    requested_revision = existing_revision or revision
    info = HfApi().dataset_info(repo_id, revision=requested_revision)
    resolved_revision = info.sha
    if existing_revision and resolved_revision != existing_revision:
        raise RuntimeError(
            f"Partial download is pinned to {existing_revision}, "
            f"not {resolved_revision}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    revision_path.write_text(resolved_revision + "\n", encoding="utf-8")
    print(f"Downloading {repo_id}@{resolved_revision} to {destination}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=resolved_revision,
        local_dir=str(destination),
        max_workers=max_workers,
    )

    arrow_datasets = sorted(
        path.parent
        for path in destination.rglob("state.json")
        if (path.parent / "dataset_info.json").is_file()
    )
    if not arrow_datasets:
        raise RuntimeError(
            f"Downloaded {repo_id}@{resolved_revision} but found no saved Arrow datasets"
        )
    return resolved_revision, len(arrow_datasets)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Real-TSF/TIME on an internet-connected preparation host."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help="Empty target directory (default: configured TIME_DATASET)",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted download in a non-empty destination",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    if args.max_workers <= 0:
        parser.error("--max-workers must be positive")

    destination = args.destination or dataset_storage_root()
    revision, count = download_time_dataset(
        destination,
        repo_id=args.repo_id,
        revision=args.revision,
        resume=args.resume,
        max_workers=args.max_workers,
    )
    print(f"Downloaded {count} TIME Arrow datasets from {args.repo_id}@{revision}")
    print(f"TIME_DATASET={destination.expanduser().resolve()}")


if __name__ == "__main__":
    main()
