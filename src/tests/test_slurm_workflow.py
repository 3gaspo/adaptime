"""Static contract for TIME's DGX/Selena foundation-model workflow."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS = (
    "chronos_bolt",
    "chronos2",
    "tirex",
    "ts_icl",
    "seasonal_naive",
)


def main() -> None:
    dgx = PROJECT_ROOT / "slurm/dgx"
    selena = PROJECT_ROOT / "slurm/selena"
    dgx_models = dgx / "foundation_models"
    selena_models = selena / "foundation_models"
    assert sorted(path.name for path in dgx.glob("*.slurm")) == [
        "foundation_summary.slurm",
    ]
    assert sorted(path.name for path in selena.glob("*.slurm")) == [
        "foundation_summary_selena.slurm",
    ]
    assert sorted(path.name for path in dgx_models.glob("*.slurm")) == [
        f"{model}.slurm" for model in sorted(MODELS)
    ]
    assert sorted(path.name for path in selena_models.glob("*.slurm")) == [
        f"{model}_selena.slurm" for model in sorted(MODELS)
    ]

    for model in MODELS:
        dgx_front = (dgx_models / f"{model}.slurm").read_text(encoding="utf-8")
        selena_front = (selena_models / f"{model}_selena.slurm").read_text(
            encoding="utf-8"
        )
        assert "#SBATCH --array" not in dgx_front
        assert "#SBATCH --array" not in selena_front
        assert "#SBATCH --partition=h100" in dgx_front
        assert "#SBATCH --partition=an" in selena_front
        assert "#SBATCH --qos=an_preemptable" in selena_front
        assert "#SBATCH --exclusive" in selena_front
        assert "#SBATCH --wckey=P12CU:DATASCIENCE" in selena_front
        assert f"/codes/{PROJECT_ROOT.name}/logs/" in selena_front
        assert f"export TIME_MODEL={model}" in dgx_front
        assert f"export TIME_MODEL={model}" in selena_front
        for front in (dgx_front, selena_front):
            assert "export PROJECT_ROOT" in front
            assert 'source "$PROJECT_ROOT/src/slurm/run_foundation_model.sh"' in front
        assert 'export TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-$HOME}"' in dgx_front
        assert 'source "$PROJECT_ROOT/src/slurm/selena_runtime.sh"' in selena_front
        assert 'TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-selena_${SLURM_JOB_ID}}"' in selena_front

    dgx_summary = (dgx / "foundation_summary.slurm").read_text(encoding="utf-8")
    selena_summary = (selena / "foundation_summary_selena.slurm").read_text(
        encoding="utf-8"
    )
    assert 'source "$PROJECT_ROOT/src/slurm/summarize_foundation_models.sh"' in dgx_summary
    assert 'source "$PROJECT_ROOT/src/slurm/summarize_foundation_models.sh"' in selena_summary
    assert "#SBATCH --partition=h100" in dgx_summary
    assert "#SBATCH --partition=an" in selena_summary
    assert "#SBATCH --qos=an_preemptable" in selena_summary
    assert "#SBATCH --exclusive" in selena_summary
    assert "#SBATCH --wckey=P12CU:DATASCIENCE" in selena_summary

    mapping = (PROJECT_ROOT / "src/slurm/foundation_model_runners.sh").read_text(
        encoding="utf-8"
    )
    assert mapping.count("    run_") == 5
    assert "FOUNDATION_MODEL_COUNT" in mapping
    for model in MODELS:
        assert f"    {model}\n" in mapping
    foundation_runners = (
        "run_chronos_bolt.sh",
        "run_chronos2.sh",
        "run_tirex.sh",
        "run_tsicl.sh",
        "run_seasonal_naive.sh",
    )
    for runner in foundation_runners:
        assert (PROJECT_ROOT / "scripts" / runner).is_file()

    workflow = (PROJECT_ROOT / "src/slurm/workflow_common.sh").read_text(
        encoding="utf-8"
    )
    for message in (
        "stage $TIME_ACTIVE_STAGE started",
        "stage $TIME_ACTIVE_STAGE completed status=success",
        "task $TIME_ACTIVE_TASK started",
        "task $TIME_ACTIVE_TASK completed status=success",
        "completed status=failed exit_code=$status",
        "completed status=success exit_code=0",
    ):
        assert message in workflow
    assert 'TIME_STATUS_ROOT="$TIME_LOGS/workflow_status/' in workflow

    model_workflow = (PROJECT_ROOT / "src/slurm/run_foundation_model.sh").read_text(
        encoding="utf-8"
    )
    assert "environment=uv" in model_workflow
    assert "TIME_MODEL:?" in model_workflow
    assert "TIME_MODEL_INDEX" not in model_workflow
    assert "SLURM_ARRAY_TASK_ID" not in model_workflow
    assert "TIMESFM_DIR" not in model_workflow

    sequential_workflow = (
        PROJECT_ROOT / "src/slurm/benchmark_foundation_models.sh"
    ).read_text(encoding="utf-8")
    assert 'for model_index in "${!FOUNDATION_MODELS[@]}"' in sequential_workflow
    assert 'TIME_MODEL="$model"' in sequential_workflow
    assert 'bash "$PROJECT_ROOT/src/slurm/run_foundation_model.sh"' in sequential_workflow
    assert 'bash "$PROJECT_ROOT/src/slurm/summarize_foundation_models.sh"' in sequential_workflow

    slurm_runner = (PROJECT_ROOT / "src/slurm/run_time_script.sh").read_text(
        encoding="utf-8"
    )
    assert 'runner_command=(uv run --no-sync bash "$run_path")' in slurm_runner
    assert 'srun --ntasks=1 "${runner_command[@]}"' in slurm_runner

    summary = (
        PROJECT_ROOT / "src/slurm/summarize_foundation_models.sh"
    ).read_text(encoding="utf-8")
    assert "uv run --no-sync python" in summary
    assert '--models "${FOUNDATION_MODELS[@]}"' in summary
    assert "reports_root" not in summary
    assert "conda" not in summary

    runtime = (PROJECT_ROOT / "src/slurm/selena_runtime.sh").read_text(
        encoding="utf-8"
    )
    common_runtime = (PROJECT_ROOT / "src/slurm/runtime_paths.sh").read_text(
        encoding="utf-8"
    )
    assert 'OUTPUTS_ROOT="${OUTPUTS_ROOT:-${TIME_OUTPUTS:-$runtime_project_root/outputs}}"' in common_runtime
    assert 'LOGS_ROOT="${LOGS_ROOT:-${TIME_LOGS:-$runtime_project_root/logs}}"' in common_runtime
    assert 'TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-/scratch/users/$selena_nni}"' in runtime
    assert 'TIME_SCRATCH_ROOT="${TIME_SCRATCH_ROOT:-$TIME_STORAGE_ROOT/codes/$PROJECT_NAME}"' in runtime
    assert 'OUTPUTS_ROOT="${OUTPUTS_ROOT:-${TIME_OUTPUTS:-$TIME_SCRATCH_ROOT/outputs}}"' in runtime
    assert 'LOGS_ROOT="${LOGS_ROOT:-${TIME_LOGS:-$TIME_SCRATCH_ROOT/logs}}"' in runtime
    assert "module load python/3.12_pypsa" in runtime
    assert "export UV_PYTHON_DOWNLOADS=never" in runtime

    submit = (PROJECT_ROOT / "scripts/submit_foundation_models.sh").read_text(
        encoding="utf-8"
    )
    assert "dgx|selena" in submit
    assert 'for model in "${FOUNDATION_MODELS[@]}"' in submit
    assert 'dependency="$(IFS=:; echo "${model_jobs[*]}")"' in submit
    assert '--dependency="afterok:$dependency"' in submit
    assert 'TIME_LAUNCH_ID=$launch_id' in submit

    code_sync = (PROJECT_ROOT / "sync_code_to_selena.sh").read_text(encoding="utf-8")
    result_sync = (PROJECT_ROOT / "sync_results_to_dgx.sh").read_text(
        encoding="utf-8"
    )
    publisher = (PROJECT_ROOT / "publish_job.sh").read_text(encoding="utf-8")
    for excluded in (
        ".git/",
        ".env",
        ".venv",
        "pyproject.toml",
        "uv.lock",
        "AGENTS.md",
        "FUTURE_WORK.md",
        "PENDING_UPDATES.md",
        "CLUSTER_STATUS.txt",
        "docs/INTERNAL_WORKFLOW.md",
        "outputs/",
        "logs/",
    ):
        assert f"--exclude='{excluded}'" in code_sync
    assert "--exclude='datasets/'" not in code_sync
    assert "--exclude='weights/'" not in code_sync
    assert "--delete-delay" in code_sync
    assert "$SCRATCH_PROJECT_ROOT/outputs/results" in code_sync
    assert "$SCRATCH_PROJECT_ROOT/logs" in code_sync
    assert "lightweight|detailed|full" in result_sync
    assert '"$SOURCE_ROOT/outputs/"' in result_sync
    assert '"$SOURCE_ROOT/logs/"' in result_sync
    assert '"$PROJECT_ROOT/outputs/selena"' in result_sync
    assert '"$PROJECT_ROOT/logs/selena"' in result_sync

    assert "lightweight|detailed|full" in publisher
    assert '. "$proxy_script"' in publisher
    assert "git pull --ff-only origin main" in publisher
    assert '"$project_root"/logs/selena/' in publisher
    assert 'find "$project_root/outputs"' in publisher
    assert "git push origin main" in publisher

    direct = (PROJECT_ROOT / "scripts/run_all_foundation_models.sh").read_text(
        encoding="utf-8"
    )
    assert 'source "$ROOT_DIR/src/slurm/foundation_model_runners.sh"' in direct
    assert 'uv run --no-sync bash "$SCRIPT_DIR/$runner"' in direct
    removed_runners = (
        "run_kairos.sh",
        "run_litespecformer.sh",
        "run_moirai.sh",
        "run_moirai2.sh",
        "run_patchtst_fm.sh",
        "run_sundial.sh",
        "run_timesfm1.sh",
        "run_timesfm2.sh",
        "run_timesfm2p5.sh",
        "run_timesfm3.sh",
        "run_toto.sh",
        "run_visiontspp.sh",
    )
    removed_experiments = (
        "kairos_model.py",
        "litespecformer_model.py",
        "moirai.py",
        "moirai2.py",
        "patchtst_fm.py",
        "sundial.py",
        "timesfm1.0.py",
        "timesfm2.0.py",
        "timesfm2.5.py",
        "timesfm3.py",
        "toto_model.py",
        "visiontspp.py",
    )
    for runner in removed_runners:
        assert not (PROJECT_ROOT / "scripts" / runner).exists()
    for experiment in removed_experiments:
        assert not (PROJECT_ROOT / "experiments" / experiment).exists()
    for runner_path in (PROJECT_ROOT / "scripts").glob("run_*.sh"):
        runner_text = runner_path.read_text(encoding="utf-8")
        assert "conda" not in runner_text.lower()
        assert "pip install" not in runner_text
        assert not any(line.startswith("srun ") for line in runner_text.splitlines())
    for runner in foundation_runners:
        runner_text = (PROJECT_ROOT / "scripts" / runner).read_text(encoding="utf-8")
        assert "set -euo pipefail" in runner_text
    print("TIME Slurm and DGX/Selena synchronization contract passed.")


if __name__ == "__main__":
    main()
