# Delta from upstream TIME

Improved TIME derives from `zqiao11/TIME` commit
`c11ed82c3eaf39e42e081e5995e7880a76f86cb9`. The initial source-only migration
consolidates reusable behavior from the former Improved repository at commit
`fd04c9f48034e2d7c604651ec23b2a177061900f`, including its final uncommitted
finite-coverage repair where that behavior belongs in the common layer.

## Runtime and lifecycle infrastructure

- Added the `TIME_DATA_ROOT`, `TIME_DATASET`, `TIME_METADATA`, `TIME_WEIGHTS`,
  `TIME_OUTPUTS`, and `TIME_LOGS` path contract and ignored local placeholders.
- Added a revision-pinned, resumable downloader and saved-Arrow validation for
  the official `Real-TSF/TIME` dataset.
- Added schema-1 task manifests, monotonic `run_n` allocation, exact-config
  skip/resume/overwrite behavior, interruption recording, explicit repeat and
  scientific-configuration selection, and selected-run pinning.
- Added strict optional and required cross-experiment compact-result reuse.
- Added compact `metrics_summary.json` artifacts with finite and total metric
  counts while retaining raw per-window metrics separately.
- Added accelerator-synchronized inference timing that excludes model loading,
  dataset construction, metric computation, and result saving.

## Model and evaluation behavior

- Retained the common `chronos_bolt`, `chronos2`, `timesfm3`, `ts_icl`, and
  `seasonal_naive` adapters and removed incompatible or unused model surfaces.
- Made learned-model loading explicit and local-only, including TS-ICL's
  no-download path and the non-shadowing `run_timesfm3.py` entry name.
- Added explicit foundation-model target-mode and covariate capability checks.
  Chronos-2 supports native multivariate targets and past-target covariates;
  TS-ICL handles target channels as univariate batch items and accepts its
  supported known covariates; unsupported combinations fail.
- Aligned covariate missingness with ordinary backbone missing-value masks.
- Normalized TS-ICL's tensor and variable-length list return forms to TIME's
  `(batch, quantile, variate, horizon)` evaluation layout.
- Made retained Python runners fail on the first unsuccessful dataset.
- Removed stochastic Seasonal Naive resampling and pass deterministic
  StatsForecast quantiles directly into TIME evaluation.
- Corrected MASE so calendar gaps remain present and only finite seasonal pairs
  enter the scale denominator.
- Suppressed only the known pandas frequency-alias deprecations and preserved
  undefined all-zero MAPE/sMAPE cells without broad warning suppression.

## Data and feature behavior

- Corrected chronological training and validation endpoints, zero-validation
  handling, combined split validation, complete-window counting, and prediction
  instance-shape validation.
- Centralized missing-history forward filling and deterministic Seasonal Naive
  point forecasting.
- Added saved-Arrow feature input, per-dataset summaries, temporal and spatial
  location/scale/frequency heterogeneity, dataset ranks, and reusable
  feature-versus-performance calculations.
- Added a shared source/window audit using prefix counts for missingness and
  adjacent-value transitions for constant-window detection.
- Repaired `seasonal_corr` handling and its quadratic pair enumeration while
  preserving variates for which the optional statistic is undefined.

## Packaging and documentation

- Removed unused or conflicting dependencies and stale packaging declarations;
  aligned package and license metadata with the Apache-2.0 project contract.
- Corrected dataset-format, feature-output, evaluation-interface, and
  prediction-archive documentation.
- Added dependency-light syntax, configuration, split, covariate, metric, and
  run-lifecycle contract checks.

## Deliberate downstream ownership

Foundation-model submission grids, Slurm files, cluster synchronization,
publication helpers, status commands, aggregate tables, feature plots, result
documents, and all log/output payloads were deliberately moved to
`evaluating_tsfms`. Improved TIME owns the reusable implementations they call,
not their scientific execution or evidence.
