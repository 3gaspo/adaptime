"""Dependency-light maintenance checks for Adaptime's TIME layer."""

from __future__ import annotations

import ast
import re
import tomllib
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


class AdaptimeMaintenanceContractTest(unittest.TestCase):
    def test_python_sources_parse(self) -> None:
        roots = [PROJECT_ROOT / "src", PROJECT_ROOT / "experiments", PROJECT_ROOT / "scripts"]
        paths = sorted(path for root in roots for path in root.rglob("*.py"))
        self.assertTrue(paths)
        for path in paths:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_configuration_files_parse(self) -> None:
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
            tomllib.load(stream)
        with (PROJECT_ROOT / "src/timebench/config/datasets.yaml").open(
            encoding="utf-8"
        ) as stream:
            config = yaml.safe_load(stream)
        self.assertIn("datasets", config)

    def test_local_document_links_exist(self) -> None:
        markdown = [PROJECT_ROOT / "README.md", *sorted((PROJECT_ROOT / "docs").glob("*.md"))]
        missing: list[str] = []
        for document in markdown:
            for target in LOCAL_LINK.findall(document.read_text(encoding="utf-8")):
                target = target.strip().split("#", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                resolved = (document.parent / target).resolve()
                if not resolved.exists():
                    missing.append(f"{document.relative_to(PROJECT_ROOT)} -> {target}")
        self.assertEqual(missing, [])

    def test_private_lifecycle_files_are_ignored(self) -> None:
        ignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        required = {
            "AGENTS.md",
            "/FUTURE_WORK.md",
            "/PENDING_UPDATES.md",
            "/CLUSTER_STATUS.txt",
            "/docs/INTERNAL_WORKFLOW.md",
        }
        self.assertTrue(required.issubset(set(ignore)))

    def test_split_boundaries_use_declared_intervals(self) -> None:
        source = (PROJECT_ROOT / "src/timebench/evaluation/data.py").read_text(encoding="utf-8")
        self.assertIn("offset=-(self._test_length + self._val_length)", source)
        self.assertIn("offset=-self._test_length", source)
        self.assertIn("math.floor(self._test_length / self.prediction_length)", source)
        self.assertIn("math.floor(self._val_length / self.prediction_length)", source)

    def test_foundation_runners_are_offline_and_fail_fast(self) -> None:
        experiments = {
            "chronos_bolt.py": ("local_files_only=True", "chronos-bolt-{model_size}"),
            "chronos2.py": (
                "BaseChronosPipeline.from_pretrained",
                "local_files_only=True",
                '"chronos2"',
            ),
            "ts_icl.py": ("allow_auto_download=False", '"tsicl/tsicl-v1.ckpt"'),
            "seasonal_naive.py": (),
        }
        for name, required in experiments.items():
            source = (PROJECT_ROOT / "experiments" / name).read_text(encoding="utf-8")
            self.assertNotIn("Failed to run experiment", source, name)
            self.assertNotIn("except Exception", source, name)
            for text in required:
                self.assertIn(text, source, name)

        runtime = (PROJECT_ROOT / "src/slurm/selena_runtime.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("export HF_HUB_OFFLINE=1", runtime)
        self.assertIn("export HF_DATASETS_OFFLINE=1", runtime)
        self.assertIn("export TRANSFORMERS_OFFLINE=1", runtime)

    def test_time_dataset_download_and_current_model_surface(self) -> None:
        downloader = (PROJECT_ROOT / "scripts/download_time_dataset.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('DEFAULT_REPO_ID = "Real-TSF/TIME"', downloader)
        self.assertIn('os.environ["HF_HUB_DISABLE_XET"] = "1"', downloader)
        self.assertLess(
            downloader.index('os.environ["HF_HUB_DISABLE_XET"] = "1"'),
            downloader.index("from huggingface_hub import"),
        )
        self.assertIn("resolved_revision = info.sha", downloader)
        self.assertIn('REVISION_FILE = ".time_snapshot_revision"', downloader)
        self.assertIn("if destination_has_files and not resume", downloader)
        self.assertIn("max_workers=max_workers", downloader)
        self.assertIn('destination.rglob("state.json")', downloader)

        self.assertFalse((PROJECT_ROOT / "experiments/tirex_model.py").exists())
        self.assertFalse((PROJECT_ROOT / "scripts/run_tirex.sh").exists())
        self.assertFalse(
            (PROJECT_ROOT / "slurm/dgx/foundation_models/tirex.slurm").exists()
        )
        self.assertFalse(
            (PROJECT_ROOT / "slurm/selena/foundation_models/tirex_selena.slurm").exists()
        )
        dependencies = tomllib.loads(
            (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]["dependencies"]
        self.assertFalse(any("tirex" in dependency.lower() for dependency in dependencies))
        registry = (PROJECT_ROOT / "src/slurm/foundation_model_runners.sh").read_text(
            encoding="utf-8"
        )
        summary = (PROJECT_ROOT / "scripts/compute_foundation_summary.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("tirex", registry.lower())
        self.assertNotIn('"tirex"', summary.lower())


if __name__ == "__main__":
    unittest.main()
