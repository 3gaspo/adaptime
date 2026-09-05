"""Dependency-free task recovery and run-selection contract."""

import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from timebench.pipeline import (
    ManifestError,
    allocate_run,
    interrupt_launch,
    load_manifest,
    select_completed_runs,
    set_selected_run,
)


def _allocate(root: Path, value: int = 1, **kwargs):
    return allocate_run(
        root,
        experiment="expe_uni",
        identity={
            "model": "model_a",
            "target_mode": "univariate",
            "dataset": "toy",
            "frequency": "H",
            "term": "short",
        },
        model_config={"value": value},
        pipeline_config={"prediction_length": 2},
        runtime_config={"device": "cpu"},
        experiment_config={"covariate_mode": "none"},
        **kwargs,
    )


def _complete(run) -> None:
    with run:
        (run.run_dir / "result.txt").write_text("complete", encoding="utf-8")
        run.complete(["result.txt"])


def main() -> None:
    previous = {
        name: os.environ.get(name)
        for name in ("TIME_LAUNCH_ID", "SLURM_JOB_ID")
    }
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "identity"
            os.environ["TIME_LAUNCH_ID"] = "launch_1"
            os.environ["SLURM_JOB_ID"] = "101"
            first = _allocate(root)
            assert first.action == "new" and first.run_dir.name == "run_0"
            _complete(first)
            component_manifest = first.run_dir / "prepared" / "manifest.json"
            component_manifest.parent.mkdir()
            component_manifest.write_text(
                '{"format": "component", "status": "completed"}',
                encoding="utf-8",
            )

            os.environ["TIME_LAUNCH_ID"] = "launch_2"
            os.environ["SLURM_JOB_ID"] = "202"
            reused = _allocate(root)
            assert reused.action == "skip" and reused.run_dir == first.run_dir
            manifest = load_manifest(reused.run_dir)
            assert manifest["launch"]["attempts"][-1]["action"] == "reuse"
            assert manifest["launch"]["attempts"][-1]["launch_id"] == "launch_2"
            assert manifest["launch"]["attempts"][-1]["slurm_job_id"] == "202"
            assert manifest["launch"]["attempts"][-1]["launched_at"]
            assert select_completed_runs(root.parent, launch_id="launch_2")[0][0] == first.run_dir
            assert interrupt_launch(root.parent, "launch_2") == []

            os.environ["TIME_LAUNCH_ID"] = "launch_3"
            interrupted = _allocate(root, force=True)
            assert interrupted.action == "overwrite"
            assert interrupt_launch(root.parent, "another_launch") == []
            assert interrupt_launch(root.parent, "launch_3") == [interrupted.run_dir]
            assert load_manifest(interrupted.run_dir)["status"] == "interrupted"

            os.environ["TIME_LAUNCH_ID"] = "launch_4"
            os.environ["SLURM_JOB_ID"] = "404"
            resumed = _allocate(root)
            assert resumed.action == "resume" and resumed.run_dir == interrupted.run_dir
            assert not (resumed.run_dir / "result.txt").exists()
            resumed_attempt = load_manifest(resumed.run_dir)["launch"]["attempts"][-1]
            assert resumed_attempt["slurm_job_id"] == "404"
            assert resumed_attempt["launched_at"]
            assert resumed_attempt["resumed_from"]["launch_id"] == "launch_3"
            assert resumed_attempt["resumed_from"]["slurm_job_id"] == "202"
            _complete(resumed)

            second_config = _allocate(root, value=2)
            assert second_config.action == "new" and second_config.run_dir.name == "run_1"
            _complete(second_config)
            try:
                select_completed_runs(root.parent)
            except ManifestError:
                pass
            else:
                raise AssertionError("ambiguous scientific configurations must fail")
            assert len(select_completed_runs(root.parent, config_policy="distinct")) == 2
            assert len(select_completed_runs(root.parent, config_policy="latest")) == 1
            assert len(select_completed_runs(root.parent, config_policy="average")) == 2

            repeat = _allocate(root, value=2, policy="new")
            assert repeat.action == "new" and repeat.run_dir.name == "run_2"
            _complete(repeat)
            selected = select_completed_runs(
                root.parent,
                config_filters={"model_config.value": 2},
                repeat_policy="selected",
            )
            assert selected[0][0] == repeat.run_dir
            set_selected_run(second_config.run_dir)
            pinned = select_completed_runs(
                root.parent,
                config_filters={"model_config.value": 2},
                repeat_policy="selected",
            )
            assert pinned[0][0] == second_config.run_dir
            assert len(
                select_completed_runs(
                    root.parent,
                    config_filters={"model_config.value": 2},
                    repeat_policy="distinct",
                )
            ) == 2
            assert len(
                select_completed_runs(
                    root.parent,
                    config_filters={"model_config.value": 2},
                    repeat_policy="average",
                )
            ) == 2

            overwritten = _allocate(root, value=3, policy="overwrite_path")
            assert overwritten.action == "overwrite"
            assert overwritten.run_dir == repeat.run_dir
            assert (overwritten.run_dir / "manifest_history").is_dir()
            assert len(load_manifest(overwritten.run_dir)["launch"]["attempts"]) == 1
            assert load_manifest(overwritten.run_dir)["project"] == PROJECT_ROOT.name
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    print("TIME task recovery and selection contract passed.")


if __name__ == "__main__":
    main()
