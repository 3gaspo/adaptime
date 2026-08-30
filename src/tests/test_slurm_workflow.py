"""Static contract for TIME's DGX/Selena foundation-model workflow."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    dgx = PROJECT_ROOT / "slurm/dgx"
    selena = PROJECT_ROOT / "slurm/selena"
    assert sorted(path.name for path in dgx.glob("*.slurm")) == [
        "foundation_models.slurm",
        "foundation_summary.slurm",
    ]
    assert sorted(path.name for path in selena.glob("*.slurm")) == [
        "foundation_models.slurm",
    ]

    dgx_model = (dgx / "foundation_models.slurm").read_text(encoding="utf-8")
    dgx_summary = (dgx / "foundation_summary.slurm").read_text(encoding="utf-8")
    selena_model = (selena / "foundation_models.slurm").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-15%4" in dgx_model
    assert "#SBATCH --array" not in selena_model
    assert "#SBATCH --partition=h100" in dgx_model
    assert "#SBATCH --partition=an" in selena_model
    assert "#SBATCH --qos=an_preemptable" in selena_model
    assert "#SBATCH --exclusive" in selena_model
    assert "#SBATCH --wckey=P12CU:DATASCIENCE" in selena_model
    assert "/codes/improved/logs/" in selena_model
    for front in (dgx_model, dgx_summary, selena_model):
        assert "export PROJECT_ROOT" in front
    for front in (dgx_model, dgx_summary):
        assert 'export TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-$HOME}"' in front
    assert 'source "$PROJECT_ROOT/src/slurm/selena_runtime.sh"' in selena_model
    assert 'source "$PROJECT_ROOT/src/slurm/benchmark_foundation_models.sh"' in selena_model
    assert 'TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-selena_${SLURM_JOB_ID}}"' in selena_model

    mapping = (PROJECT_ROOT / "src/slurm/foundation_model_runners.sh").read_text(
        encoding="utf-8"
    )
    assert mapping.count("    run_") == 16
    assert "FOUNDATION_MODEL_COUNT" in mapping
    for runner in (
        "run_chronos_bolt.sh",
        "run_chronos2.sh",
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
        "run_tirex.sh",
        "run_toto.sh",
        "run_tsicl.sh",
        "run_visiontspp.sh",
    ):
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
    assert 'export ENV_NAME="${TIME_ENV_NAME:-time_${model}}"' in model_workflow
    assert "TIME_MODEL_INDEX" in model_workflow
    assert "SLURM_ARRAY_TASK_ID" in model_workflow
    assert "timesfm_${model}" in model_workflow

    selena_workflow = (
        PROJECT_ROOT / "src/slurm/benchmark_foundation_models.sh"
    ).read_text(encoding="utf-8")
    assert 'model_indices=("${!FOUNDATION_MODELS[@]}")' in selena_workflow
    assert 'bash "$PROJECT_ROOT/src/slurm/run_foundation_model.sh"' in selena_workflow
    assert 'bash "$PROJECT_ROOT/src/slurm/summarize_foundation_models.sh"' in selena_workflow

    slurm_runner = (PROJECT_ROOT / "src/slurm/run_time_script.sh").read_text(
        encoding="utf-8"
    )
    assert 'srun --ntasks=1 bash "$run_path"' in slurm_runner

    runtime = (PROJECT_ROOT / "src/slurm/selena_runtime.sh").read_text(
        encoding="utf-8"
    )
    common_runtime = (PROJECT_ROOT / "src/slurm/runtime_paths.sh").read_text(
        encoding="utf-8"
    )
    assert 'TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-$runtime_project_root}"' in common_runtime
    assert 'TIME_DATA_ROOT="${TIME_DATA_ROOT:-$TIME_STORAGE_ROOT/datasets}"' in common_runtime
    assert 'TIME_WEIGHTS="${TIME_WEIGHTS:-$TIME_STORAGE_ROOT/weights}"' in common_runtime
    assert 'TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-/scratch/users/$selena_nni}"' in runtime
    assert 'TIME_SCRATCH_ROOT="${TIME_SCRATCH_ROOT:-$TIME_STORAGE_ROOT/codes/$PROJECT_NAME}"' in runtime
    assert 'TIME_DATA_ROOT="${TIME_DATA_ROOT:-$TIME_STORAGE_ROOT/datasets}"' in runtime
    assert 'TIME_WEIGHTS="${TIME_WEIGHTS:-$TIME_STORAGE_ROOT/weights}"' in runtime
    assert 'TIME_OUTPUTS="${TIME_OUTPUTS:-$TIME_SCRATCH_ROOT/outputs}"' in runtime
    assert 'TIME_LOGS="${TIME_LOGS:-$TIME_SCRATCH_ROOT/logs}"' in runtime

    submit = (PROJECT_ROOT / "scripts/submit_foundation_models.sh").read_text(
        encoding="utf-8"
    )
    assert 'dependency="afterok:$evaluation_job"' in submit
    assert 'TIME_LAUNCH_ID=$launch_id' in submit
    assert "Selena: submit slurm/selena/foundation_models.slurm directly" in submit
    assert 'TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-$HOME}"' in submit
    assert "dgx|selena" not in submit

    code_sync = (PROJECT_ROOT / "sync_code_to_selena.sh").read_text(encoding="utf-8")
    result_sync = (PROJECT_ROOT / "sync_results_to_dgx.sh").read_text(
        encoding="utf-8"
    )
    publisher = (PROJECT_ROOT / "publish_job.sh").read_text(encoding="utf-8")
    for excluded in (
        ".git/",
        ".env",
        "AGENTS.md",
        "FUTURE_WORK.md",
        "PENDING_UPDATES.md",
        "CLUSTER_STATUS.txt",
        "docs/INTERNAL_WORKFLOW.md",
        "datasets/",
        "weights/",
        "outputs/",
        "logs/",
    ):
        assert f"--exclude='{excluded}'" in code_sync
    assert "--delete-delay" in code_sync
    assert "$SCRATCH_PROJECT_ROOT/outputs/results" in code_sync
    assert "$SCRATCH_PROJECT_ROOT/logs" in code_sync
    assert 'SCRATCH_STORAGE_ROOT="${TIME_SELENA_STORAGE_ROOT:-/scratch/users/$nni}"' in code_sync
    assert "'$SCRATCH_STORAGE_ROOT/datasets'" in code_sync
    assert "'$SCRATCH_STORAGE_ROOT/weights'" in code_sync
    assert "outputs_selena" not in code_sync
    assert "logs_selena" not in code_sync
    assert "lightweight|detailed|full" in result_sync
    assert "config.json" in result_sync
    assert "metrics.npz" in result_sync
    assert '"$SOURCE_ROOT/outputs/"' in result_sync
    assert '"$SOURCE_ROOT/logs/"' in result_sync
    assert '"$PROJECT_ROOT/outputs/selena"' in result_sync
    assert '"$PROJECT_ROOT/logs/selena"' in result_sync
    assert "outputs_selena" not in result_sync
    assert "logs_selena" not in result_sync

    assert "lightweight|detailed|full" in publisher
    assert '. "$proxy_script"' in publisher
    assert "git pull --ff-only origin main" in publisher
    assert '"$project_root"/logs/selena/' in publisher
    assert "foundation_model_summary.csv" in publisher
    assert "config.json" in publisher
    assert "metrics.npz" in publisher
    assert "git push origin main" in publisher

    direct = (PROJECT_ROOT / "scripts/run_all_foundation_models.sh").read_text(
        encoding="utf-8"
    )
    assert 'source "$ROOT_DIR/src/slurm/foundation_model_runners.sh"' in direct
    assert not (PROJECT_ROOT / "outputs_selena").exists()
    assert not (PROJECT_ROOT / "logs_selena").exists()
    print("TIME Slurm and DGX/Selena synchronization contract passed.")


if __name__ == "__main__":
    main()
