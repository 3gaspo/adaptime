# Code architecture

Adaptime keeps TIME's inherited foundation benchmark recognizable while
isolating the proposal pipeline beneath `src/timebench/`.

The inherited baseline path is:

```text
scripts/run_*.sh
  -> experiments/*.py
  -> timebench.evaluation.Dataset and model inference
  -> timebench.pipeline.allocate_run
  -> predictions, metrics, completed schema-1 manifest
  -> scripts/compute_foundation_summary.py
```

The Adaptime path is:

```text
TIME saved-Arrow dataset + dataset YAML
  -> evaluation/adaptation_data.py: fixed datastore and pre-test/test indices
  -> pipeline/adaptime_extraction.py: V, C, neighbors, Y, N
  -> pipeline/adaptime_training.py: train statistics and validation selection
  -> pipeline/adaptime_testing.py: frozen V/C/Adaptime TIME comparison
```

`timebench.adaptime` owns exact retrieval and the proposal's readable ridge
math. `timebench.model_loading` owns foundation construction, capability
declarations, offline checkpoints, and the tensor/covariate adapters.
`timebench.evaluation.adaptation_data` owns chronological windows and lazy
Arrow access. `timebench.pipeline` owns disk-backed extraction, model/result
manifests, and TIME-wide orchestration. `timebench.scripts` exposes the Python
commands. `src/slurm` owns shared DGX/Selena workflow implementations, while
`slurm` contains concise submit-ready fronts.

The inherited `evaluation/window_audit.py` diagnostics inspect the exact TIME
test queries and model-effective context limits. Their cluster workflow also
uses `feature/features_runner.py --force` to refresh full-series statistics;
compact and detailed diagnostics are routed through the standard result-sync
and publication scopes.

The extraction boundary is deliberate. Large values remain in Arrow or `.npy`
memory maps; distance matrices are bounded by query and datastore blocks;
neighbor foundation forecasts are computed once per selected datastore row;
and every K/alpha candidate reuses the same extraction. Ridge fitting streams
float64 sufficient statistics, so it never materializes the flattened design.

The current proposal path is univariate. Native multivariate evaluation remains
an inherited Chronos-2 control and is not mixed into `full_ridge_shared`.
