"""Static contract for Adaptime's Ridge/TS-RAG workflows and inherited runtime."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    dgx = PROJECT_ROOT / "slurm/dgx"
    selena = PROJECT_ROOT / "slurm/selena"
    # Deliberate downstream deletions survive parent merges.
    for name in (
        "submit_foundation_models.sh",
        "run_all_foundation_models.sh",
        "channels_comparison.sh",
        "run_chronos2_comparison.sh",
        "dataset_diagnostics.sh",
        "download_time_dataset.py",
        "audit_time_windows.py",
        "compute_foundation_summary.py",
        "compute_local_leaderboard.py",
        "plot_feature_performance.py",
        "run_chronos2.sh",
        "run_chronos_bolt.sh",
        "run_timesfm3.sh",
        "run_tsicl.sh",
    ):
        assert not (PROJECT_ROOT / "scripts" / name).exists(), name
    for name in (
        "foundation_model_schedule.sh",
        "benchmark_foundation_models.sh",
        "summarize_foundation_models.sh",
        "run_chronos2_comparison.sh",
        "run_dataset_diagnostics.sh",
        "export_dataset_metadata.sh",
    ):
        assert not (PROJECT_ROOT / "src/slurm" / name).exists(), name
    for cluster in (dgx, selena):
        assert not list((cluster / "chronos2_comparison").glob("*.slurm"))
        assert sorted(path.name for path in (cluster / "foundation_models").glob("*.slurm")) == [
            "seasonal_naive.slurm" if cluster == dgx else "seasonal_naive_selena.slurm"
        ]
    assert sorted(path.name for path in dgx.glob("*.slurm")) == [
        "adaptime_comparison.slurm", "time_inference.slurm", "tsrag_comparison.slurm"
    ]
    assert sorted(path.name for path in selena.glob("*.slurm")) == [
        "adaptime_comparison_selena.slurm", "time_inference.slurm", "tsrag_comparison_selena.slurm"
    ]

    adaptime_fronts = (
        (dgx / "adaptime_comparison.slurm").read_text(encoding="utf-8"),
        (selena / "adaptime_comparison_selena.slurm").read_text(encoding="utf-8"),
    )
    for front in adaptime_fronts:
        assert "#SBATCH --array" not in front
        assert 'source "$PROJECT_ROOT/src/slurm/run_adaptime_comparison.sh"' in front
    assert "#SBATCH --partition=h100" in adaptime_fronts[0]
    assert "#SBATCH --partition=an" in adaptime_fronts[1]
    assert "#SBATCH --qos=an_preemptable" in adaptime_fronts[1]
    assert "#SBATCH --exclusive" in adaptime_fronts[1]
    assert "#SBATCH --wckey=P12CU:DATASCIENCE" in adaptime_fronts[1]

    adaptime_workflow = (
        PROJECT_ROOT / "src/slurm/run_adaptime_comparison.sh"
    ).read_text(encoding="utf-8")
    assert "for stage in extract train test" not in adaptime_workflow
    assert "uv run --no-sync python -m timebench.scripts.run_adaptation_stage" in adaptime_workflow
    assert "ADAPTIME_K_VALUES:-1 5 10 15" in adaptime_workflow
    assert "ADAPTIME_ALPHA_VALUES:-0.001 0.01 0.1" in adaptime_workflow
    assert "ADAPTIME_MINIMUM_QUERY_FINITE_FRACTION:-0.8" in adaptime_workflow
    assert '--stage "$ADAPTIME_STAGE_VALUE"' in adaptime_workflow
    assert '--method "$ADAPTIME_METHOD_VALUE"' in adaptime_workflow
    assert 'TIME_RESULT_SCOPE="$ADAPTIME_OUTPUT_ROOT_VALUE"' in adaptime_workflow
    assert (
        "ADAPTIME_EXCLUDE_DATASETS:-Coastal_T_S/5T,current_velocity/20T,"
        "azure2019_D/5T,azure2019_I/5T"
    ) in adaptime_workflow
    assert '--exclude-datasets "$ADAPTIME_EXCLUDE_DATASETS_VALUE"' in adaptime_workflow
    assert 'time_stage_start "$ADAPTIME_STAGE_VALUE"' in adaptime_workflow

    tsrag_fronts = (
        (dgx / "tsrag_comparison.slurm").read_text(encoding="utf-8"),
        (selena / "tsrag_comparison_selena.slurm").read_text(encoding="utf-8"),
    )
    for front in tsrag_fronts:
        assert "export ADAPTIME_METHOD=tsrag" in front
        assert 'source "$PROJECT_ROOT/src/slurm/run_adaptime_comparison.sh"' in front
    assert not (PROJECT_ROOT / "src/slurm/run_tsrag_comparison.sh").exists()

    adaptime_submit = (
        PROJECT_ROOT / "scripts/submit_adaptime_comparison.sh"
    ).read_text(encoding="utf-8")
    assert "dgx|selena" in adaptime_submit
    assert "adaptime_comparison.slurm" in adaptime_submit
    assert "adaptime_comparison_selena.slurm" in adaptime_submit
    assert "ADAPTIME_STAGE=prepare" in adaptime_submit
    assert "ADAPTIME_METHOD=ridge,ADAPTIME_STAGE=pipeline" in adaptime_submit
    assert "ADAPTIME_METHOD=tsrag,ADAPTIME_STAGE=pipeline" in adaptime_submit
    assert "ADAPTIME_METHOD=seasonal_naive,ADAPTIME_STAGE=pipeline" not in adaptime_submit
    assert "ADAPTIME_METHOD=unified,ADAPTIME_STAGE=report" in adaptime_submit
    assert 'ridge_job="$(sbatch' in adaptime_submit
    assert '--dependency="afterok:$ridge_job:$tsrag_job"' in adaptime_submit
    assert "prepare=%s vanilla=%s ridge=%s tsrag=%s report=%s seasonal=shared" in adaptime_submit
    assert 'if [ -n "${ADAPTIME_RIDGE_RESULTS_PATH:-}" ]' not in adaptime_submit

    tsrag_submit = (
        PROJECT_ROOT / "scripts/submit_tsrag_comparison.sh"
    ).read_text(encoding="utf-8")
    assert "ADAPTIME_STAGE=pipeline" in tsrag_submit
    assert 'if [ -n "${ADAPTIME_RIDGE_RESULTS_PATH:-}" ]' in tsrag_submit

    result_sync = (PROJECT_ROOT / "sync_results_to_dgx.sh").read_text(
        encoding="utf-8"
    )
    publisher = (PROJECT_ROOT / "publish_job.sh").read_text(encoding="utf-8")
    for compact_artifact in (
        "model_manifest.json",
        "result_manifest.json",
        "selection.json",
        "comparison_summary.json",
        "time_summary_manifest.json",
        "time_summary.json",
        "time_tasks.csv",
        "report_manifest.json",
        "comparison.csv",
    ):
        assert compact_artifact in result_sync
        assert compact_artifact in publisher

    workflow_source = (
        PROJECT_ROOT / "src/timebench/pipeline/adaptime_workflow.py"
    ).read_text(encoding="utf-8")
    assert '"evaluator": "timebench.evaluation.saver.save_window_predictions"' in workflow_source
    assert "_matching_evaluation_runs(" in workflow_source
    assert "pipeline_stages = {" in workflow_source
    assert "allocate_run(" in workflow_source
    assert "if not run.should_run:" in workflow_source
    assert "with run:" in workflow_source
    assert "build_adaptation_comparison(" in workflow_source

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

    slurm_runner = (PROJECT_ROOT / "src/slurm/run_time_script.sh").read_text(
        encoding="utf-8"
    )
    assert 'runner_command=(uv run --no-sync bash "$run_path")' in slurm_runner
    assert 'srun --ntasks=1 "${runner_command[@]}"' in slurm_runner
    assert 'if [ ! -d "$TIME_DATASET" ]' in slurm_runner
    assert 'TIME dataset directory not found: $TIME_DATASET' in slurm_runner

    runtime = (PROJECT_ROOT / "src/slurm/selena_runtime.sh").read_text(
        encoding="utf-8"
    )
    common_runtime = (PROJECT_ROOT / "src/slurm/runtime_paths.sh").read_text(
        encoding="utf-8"
    )
    assert 'OUTPUTS_ROOT="${OUTPUTS_ROOT:-${TIME_OUTPUTS:-$runtime_project_root/outputs}}"' in common_runtime
    assert 'LOGS_ROOT="${LOGS_ROOT:-${TIME_LOGS:-$runtime_project_root/logs}}"' in common_runtime
    assert 'TIME_METADATA="${TIME_METADATA:-$TIME_DATA_ROOT/time_metadata}"' in common_runtime
    runtime_mkdir = next(
        line for line in common_runtime.splitlines() if line.startswith("mkdir -p ")
    )
    assert '"$TIME_DATASET"' not in runtime_mkdir
    assert 'TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-/scratch/users/$selena_nni}"' in runtime
    assert 'TIME_SCRATCH_ROOT="${TIME_SCRATCH_ROOT:-$TIME_STORAGE_ROOT/codes/$PROJECT_NAME}"' in runtime
    assert 'OUTPUTS_ROOT="${OUTPUTS_ROOT:-${TIME_OUTPUTS:-$TIME_SCRATCH_ROOT/outputs}}"' in runtime
    assert 'LOGS_ROOT="${LOGS_ROOT:-${TIME_LOGS:-$TIME_SCRATCH_ROOT/logs}}"' in runtime
    assert 'TIME_METADATA="${TIME_METADATA:-$TIME_DATA_ROOT/time_metadata}"' in runtime
    assert "module load python/3.12_pypsa" in runtime
    assert "export UV_PYTHON_DOWNLOADS=never" in runtime
    assert "export HF_HUB_OFFLINE=1" in runtime
    assert "export HF_DATASETS_OFFLINE=1" in runtime
    assert "export TRANSFORMERS_OFFLINE=1" in runtime
    assert runtime.index('source "$PROJECT_ROOT/.env"') < runtime.index(
        'TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-/scratch/users/$selena_nni}"'
    )
    assert 'if [[ -v "$runtime_path_variable" ]]' in runtime
    assert (
        'runtime_path_overrides["$runtime_path_variable"]="${!runtime_path_variable}"'
        in runtime
    )

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
    assert "$SCRATCH_PROJECT_ROOT/outputs" in code_sync
    assert "$SCRATCH_PROJECT_ROOT/logs" in code_sync
    assert "lightweight|detailed|full" in result_sync
    assert '"$SOURCE_ROOT/outputs/"' in result_sync
    assert '"$SOURCE_ROOT/logs/"' in result_sync
    assert '"$PROJECT_ROOT/outputs/selena"' in result_sync
    assert '"$PROJECT_ROOT/logs/selena"' in result_sync
    assert '"--include=/dataset_metadata/$JOB_ID/***"' in result_sync
    assert "--include=mase_vs_features.svg" in result_sync
    assert "--include=mase_vs_features_data.csv" in result_sync
    assert "--include=mase_vs_features_correlations.csv" in result_sync
    assert "--include=SELECTED_RUNS.json" in result_sync
    assert "--include=*/manifest_history/*.json" in result_sync

    assert "lightweight|detailed|full" in publisher
    assert '. "$proxy_script"' in publisher
    assert "git pull --ff-only origin main" in publisher
    assert '"$project_root"/logs/selena/' in publisher
    assert 'find "$project_root/outputs"' in publisher
    assert "logs/selena/dataset_metadata" in publisher
    assert "-name mase_vs_features.svg" in publisher
    assert "-name mase_vs_features_data.csv" in publisher
    assert "-name mase_vs_features_correlations.csv" in publisher
    assert "-name SELECTED_RUNS.json" in publisher
    assert "-path '*/manifest_history/*.json'" in publisher
    assert "git push origin main" in publisher

    lifecycle = (PROJECT_ROOT / "src/timebench/pipeline/runs.py").read_text(
        encoding="utf-8"
    )
    assert 'CONFLICT_POLICIES = ("overwrite_exact", "overwrite_path", "new")' in lifecycle
    assert 'CONFIG_POLICIES = ("error", "distinct", "latest", "average")' in lifecycle
    assert 'REPEAT_POLICIES = ("selected", "latest", "distinct", "average")' in lifecycle
    assert '"slurm_job_id": os.environ.get("SLURM_JOB_ID")' in lifecycle
    assert '"launched_at": launched_at' in lifecycle
    assert (PROJECT_ROOT / "scripts/select_result_run.py").is_file()
    assert (PROJECT_ROOT / "scripts/interrupt_result_launch.py").is_file()

    print("Adaptime workflow ownership and inherited runtime contracts passed.")


if __name__ == "__main__":
    main()
