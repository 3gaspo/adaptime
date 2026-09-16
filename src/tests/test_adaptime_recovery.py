"""Dependency-free regression of the real recovery control flow and reports.

Execute the workflow functions from their AST to avoid cluster-only imports;
use real lifecycle and reporting modules with tiny synthetic saver artifacts.
"""

from __future__ import annotations

import ast
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/timebench/pipeline/adaptime_workflow.py"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runs = load_file("timebench.pipeline.runs", ROOT / "src/timebench/pipeline/runs.py")
reports = load_file("recovery_reports", ROOT / "src/timebench/results/adaptation.py")
methods_tree = ast.parse((ROOT / "src/timebench/pipeline/adaptation_prediction.py").read_text(encoding="utf-8"))
METHODS = next(ast.literal_eval(node.value) for node in methods_tree.body
               if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "ADAPTATION_METHODS" for target in node.targets))
FUNCTIONS = {
    "_allocation", "_run_evaluation", "_run_ridge_fallback", "_ridge_evaluations_complete",
    "_run_ridge_task",
    "run_adaptation_stage",
}
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
code = compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
                          + [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS],
                          type_ignores=[])), str(SOURCE), "exec")


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.task = types.SimpleNamespace(dataset="tiny/D", term="short", preparation=types.SimpleNamespace(prediction_length=1))
        self.workflow = types.SimpleNamespace(validate=lambda method: None, target_mode="univariate")
        self.saved = []
        self.failure_method = None

        def spec(task, workflow, stage, method):
            return ("adaptime_evaluation", {"model": method, "dataset": task.dataset.rsplit("/", 1)[0], "frequency": "D", "term": task.term, "target_mode": "univariate"},
                    {"method": method}, {}, {"phase": "evaluation"})

        def stage_root(root, stage, method, task):
            return root / stage / method / task.dataset / task.term

        def completed(root, scientific):
            selected = [path.parent for path in root.glob("run_*/manifest.json")
                        if json.loads(path.read_text())["status"] == "completed"]
            if len(selected) != 1:
                raise FileNotFoundError(root)
            return selected[0]

        def data(root, task, workflow):
            path = root / "prepared" / task.dataset / task.term / "manifest.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"counts": {"test": 1}, "dataset": task.dataset, "term": task.term}))
            return path

        def saver(prepared, prediction, output, *, method, evaluation_grid_path):
            if self.failure_method is not None and method == self.failure_method:
                raise ValueError("invalid Ridge prediction")
            self.saved.append((method, prediction))
            data_config = json.loads(prepared.read_text())
            dataset_config = f"{data_config['dataset']}/{data_config['term']}"
            config = {"dataset_config": dataset_config, "num_series": 1, "num_windows": 1, "num_variates": 1,
                      "prediction_length": 1, "seasonality": 1, "target_mode": "univariate",
                      "evaluation_grid": {"valid_values": 1, "total_values": 1},
                      "selected_adaptation": {"method": "full_ridge_shared"},
                      "fallback_to_vanilla": {"count": 0, "eligible_evaluation_windows": 1, "rate": 0.0}}
            summary = {"dataset_config": dataset_config, "metrics": {"MASE": {"mean": 1.0 if method == "vanilla" else 2.0,
                       "finite_values": 1, "evaluation_values": 1, "total_values": 1}}, "inference_seconds": 3.0, "model": method}
            for name, value in (("config.json", config), ("metrics_summary.json", summary)):
                (output / name).write_text(json.dumps(value))
            for name in ("predictions.npz", "metrics.npz"):
                (output / name).write_bytes(b"synthetic raw artifact")

        self.env = {"Path": Path, "json": json, "os": os, "csv": csv,
                    "ADAPTATION_METHODS": METHODS, "allocate_run": runs.allocate_run,
                    "_spec": spec, "_stage_root": stage_root, "_completed_run": completed,
                    "_data_manifest": data, "_vanilla_manifest": lambda root, task, workflow: root / "vanilla_source/manifest.json",
                    "_artifact_manifest": lambda *args: self.root / "ridge_source/prediction_manifest.json",
                    "resolve_shared_evaluation_grid": lambda *args: self.root / "grid.npz",
                    "evaluate_point_predictions": saver, "interrupt_launch": runs.interrupt_launch,
                    "_run_vanilla": lambda root, task, workflow: root / "vanilla_source/manifest.json",
                    "_run_prepare": data, "load_dataset_config": lambda path: {},
                    "workflow_tasks": lambda *args: [self.task],
                    "outputs_root": lambda: self.root, "build_adaptation_comparison": reports.build_adaptation_comparison,
                    "METHODS": ("ridge", "vanilla", "unified", "tsrag", "seasonal_naive", "rolling_y_ridge_horizon"),
                    "STAGES": ("prepare", "vanilla", "extract", "fit", "extract_eval", "predict", "evaluate", "report", "pipeline", "all"),
                    "ROLLING_RIDGE_METHOD": "rolling_y_ridge_horizon"}
        exec(code, self.env)
        for name in ("_run_ridge_extraction", "_run_fit", "_run_ridge_eval_extraction", "_run_ridge_prediction"):
            self.env[name] = lambda *args: self.root / "complete_phase"

    def tearDown(self):
        self.temp.cleanup()

    def configs(self):
        return {method: json.loads(next((self.root / "evaluations" / method).rglob("config.json")).read_text())
                for method in METHODS}

    def seed(self):
        return self.env["_run_ridge_fallback"](self.root, self.task, self.workflow, {"code": "ridge_not_completed"})

    def test_baseline_routing_and_reference_outputs(self):
        outputs = self.seed()
        self.assertEqual(self.saved, [("vanilla", self.root / "vanilla_source/manifest.json")])
        self.assertFalse(self.env["_ridge_evaluations_complete"](self.root, self.task, self.workflow))
        for output in outputs:
            config = json.loads((output.parent / "config.json").read_text())
            self.assertEqual(config["fallback_to_vanilla"]["rate"], 1.0)
            self.assertEqual(json.loads(output.read_text())["metrics"]["MASE"]["mean"], 1.0)
            manifest = json.loads((output.parent / "manifest.json").read_text())
            for relative in manifest["required_artifacts"]:
                self.assertTrue((output.parent / relative).is_file())
            self.assertFalse((output.parent / "metrics.npz").exists())  # references, no large duplicate payload

    def test_errors_in_every_ridge_phase_keep_outputs(self):
        phases = ("extract", "fit", "extract_eval", "predict", "evaluate")
        for index, stage in enumerate(phases):
            with self.subTest(stage=stage):
                self.root = Path(self.temp.name) / stage
                self.root.mkdir()
                self.seed()
                mapping = {"extract": "_run_ridge_extraction", "fit": "_run_fit", "extract_eval": "_run_ridge_eval_extraction", "predict": "_run_ridge_prediction"}
                calls = []
                for phase, name in mapping.items():
                    def runner(*args, phase=phase):
                        calls.append(phase)
                        if phase == stage:
                            raise MemoryError("synthetic failed Ridge stage")
                        return self.root / "complete_phase"
                    self.env[name] = runner
                self.failure_method = "full_ridge_shared" if stage == "evaluate" else None
                self.env["_run_ridge_task"](self.root, self.task, self.workflow)
                self.assertEqual(calls, list(phases[:min(index + 1, 4)]))
                self.assertTrue(self.env["_ridge_evaluations_complete"](self.root, self.task, self.workflow))
                config = self.configs()["full_ridge_shared"]
                self.assertEqual(config["adaptation_fallback_reason"]["code"], "ridge_stage_failed")
                self.assertEqual(config["fallback_to_vanilla"]["rate"], 1.0)

    def test_completed_predictions_replace_provisional_fallback(self):
        self.seed()
        self.env["_run_ridge_task"](self.root, self.task, self.workflow)
        config = self.configs()["full_ridge_shared"]
        self.assertIsNone(config.get("adaptation_fallback_reason"))
        summary = next((self.root / "evaluations/full_ridge_shared").rglob("metrics_summary.json"))
        self.assertEqual(json.loads(summary.read_text())["metrics"]["MASE"]["mean"], 2.0)

    def test_launcher_paths_and_cli_agree(self):
        for path in (SOURCE, ROOT / "src/timebench/scripts/run_adaptation_stage.py", ROOT / "src/timebench/evaluation/adaptation.py"):
            ast.parse(path.read_text(encoding="utf-8"))
        shell = (ROOT / "src/slurm/run_adaptime_comparison.sh").read_text(encoding="utf-8")
        self.assertLess(shell.index('export OUTPUTS_ROOT="$ADAPTIME_PROJECT_ROOT/outputs"'), shell.index('source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"'))
        self.assertIn('ADAPTIME_PROJECT_ROOT="$TIME_STORAGE_ROOT/codes/adaptime"', shell)
        self.assertIn('export ADAPTIME_OUTPUT_ROOT="$OUTPUTS_ROOT/adaptime"', shell)
        for path in (SOURCE, ROOT / "src/timebench/scripts/run_adaptation_stage.py", ROOT / "src/slurm/run_adaptime_comparison.sh"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("quick_results", text)
            self.assertNotIn("ridge_time_budget", text)

    def test_normal_pipeline_continues_after_failed_fit(self):
        healthy = types.SimpleNamespace(dataset="healthy/D", term="short", preparation=self.task.preparation)
        self.env["workflow_tasks"] = lambda *args: [self.task, healthy]
        fitted = []
        def fit(root, task, workflow):
            fitted.append(task.dataset)
            if task.dataset == "tiny/D":
                raise ValueError("cannot score empty ridge statistics")
            return root / "healthy_fit"
        self.env["_run_fit"] = fit
        self.env["run_adaptation_stage"]("pipeline", "ridge", self.workflow, output_root=self.root)
        self.assertEqual(fitted, ["tiny/D", "healthy/D"])
        for task in (self.task, healthy):
            self.assertTrue(self.env["_ridge_evaluations_complete"](self.root, task, self.workflow))
        bad = json.loads((self.root / "evaluations/full_ridge_shared/tiny/D/short/run_0/config.json").read_text())
        good = json.loads((self.root / "evaluations/full_ridge_shared/healthy/D/short/run_0/config.json").read_text())
        self.assertEqual(bad["fallback_to_vanilla"]["rate"], 1.0)
        self.assertIsNone(good.get("adaptation_fallback_reason"))

    def test_fallback_outputs_join_the_real_comparison(self):
        self.seed()
        for method in ("seasonal_naive", "tsrag"):
            self.env["_run_evaluation"](self.root, self.task, self.workflow, method)
        report = reports.build_adaptation_comparison(
            method_results_roots={method: self.root / "evaluations" / method for method in (*METHODS, "seasonal_naive", "tsrag")},
            output_dir=self.root / "report", expected_tasks=[("tiny/D", "short")],
        )
        with (report.parent / "comparison.csv").open(newline="") as stream:
            cells = {row["base_method"]: row for row in csv.DictReader(stream)}
        self.assertEqual(cells["full_ridge_shared"]["MASE"], cells["vanilla"]["MASE"])
        self.assertEqual(cells["full_ridge_shared"]["inference_seconds"], cells["vanilla"]["inference_seconds"])
        self.assertEqual(cells["full_ridge_shared"]["task_fallback"], "True")
        with (report.parent / "adaptation_summary.csv").open(newline="") as stream:
            summary = {row["base_model"]: row for row in csv.DictReader(stream)}
        self.assertEqual(float(summary["full_ridge_shared"]["MASE"]), 1.0)
        self.assertEqual(float(summary["full_ridge_shared"]["inference_seconds"]), 3.0)
        self.assertEqual(int(summary["full_ridge_shared"]["task_fallbacks"]), 1)


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTest)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
