# Code architecture

This document owns Adaptime's public source-responsibility map and executable
data flow.

Adaptime currently retains TIME's recognizable benchmark layout. The inherited
execution path is:

```text
scripts/run_*.sh
  -> experiments/*.py
  -> timebench.evaluation.Dataset and model inference
  -> timebench.pipeline.allocate_run
  -> predictions, metrics, and a completed schema-1 manifest
  -> scripts/compute_foundation_summary.py
```

`src/timebench/evaluation/` owns TIME window construction, covariate validation,
prediction saving, and metrics. `src/timebench/pipeline/` owns run allocation
and manifest-based result selection. `src/timebench/feature/` owns saved-Arrow
feature extraction and feature-versus-performance analysis. `src/slurm/` owns
the shared DGX/Selena workflow implementation, while `slurm/` contains the
submit-ready fronts.

The five inherited model adapters remain under `experiments/`. Chronos-2 owns
the native-multivariate and covariate-capable paths; other baseline adapters
declare their narrower capabilities and reject unsupported inputs.

Adaptime's retrieval, adaptor-training, and proposed-method owners have not yet
been fixed, so no custom package boundary or runtime stage is asserted here.

When the implementation stabilizes, this page will identify each source owner,
extend this map from pre-test datasets and query windows through retrieval,
adaptation, prediction, and metrics, and state the boundaries between inherited
TIME code, adapted external methods, and the Adaptime proposal.
