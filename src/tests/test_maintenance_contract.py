"""Dependency-light maintenance checks for the Improved TIME layer."""

from __future__ import annotations

import ast
import re
import tomllib
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


class ImprovedMaintenanceContractTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
