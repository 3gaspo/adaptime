#!/usr/bin/env python3
"""Download the official TIME Arrow dataset for offline cluster evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from timebench.paths import dataset_storage_root


DEFAULT_REPO_ID = "Real-TSF/TIME"


def download_time_dataset(
    destination: Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = "main",
) -> tuple[str, int]:
    """Download one immutable repository revision and validate saved Arrow datasets."""
    destination = destination.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"TIME destination is not empty: {destination}. "
            "Choose an empty directory so stale datasets cannot survive an update."
        )

    info = HfApi().dataset_info(repo_id, revision=revision)
    resolved_revision = info.sha
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=resolved_revision,
        local_dir=str(destination),
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
    args = parser.parse_args()

    destination = args.destination or dataset_storage_root()
    revision, count = download_time_dataset(
        destination,
        repo_id=args.repo_id,
        revision=args.revision,
    )
    print(f"Downloaded {count} TIME Arrow datasets from {args.repo_id}@{revision}")
    print(f"TIME_DATASET={destination.expanduser().resolve()}")


if __name__ == "__main__":
    main()
