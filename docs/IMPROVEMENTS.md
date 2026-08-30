# Delta from upstream TIME

Improved currently derives from `zqiao11/TIME` commit
`c11ed82c3eaf39e42e081e5995e7880a76f86cb9`. This page records intentional
reusable divergences so later upstream merges can preserve or retire them
deliberately.

## Runtime infrastructure

- Added one environment-variable path contract and local artifact
  placeholders for datasets, weights, outputs, and logs.
- Routed every model runner's default results below `TIME_OUTPUTS/results` and
  VisionTS++ checkpoints below `TIME_WEIGHTS`.
- Added shared Hugging Face/Torch cache defaults and a local cluster wrapper
  that calls the existing TIME run scripts with externally configured paths.
- Completed the previously headerless Seasonal Naive shell runner with an
  executable environment setup and project-root resolution.
- Added one accelerator-synchronized test-loop timer used by every model
  runner. Each task stores total inference seconds in `config.json`, excluding
  model loading, dataset construction, metric computation, and result saving.
- Added an all-foundation-model runner and CSV/Markdown summary whose MASE
  macro-average weights H settings equally within dataset/frequency entries
  and then weights those entries equally; timing totals require complete task
  coverage.
- Added a DGX Slurm array with one isolated job per foundation-model runner and
  a dependent summary job. Added a TSFM-style Selena Slurm front submitted
  directly with `sbatch`; its single allocation runs all model tasks and the
  final summary sequentially without a Bash submission wrapper. Both paths
  emit explicit stage/task/workflow completion records and durable status
  files below the configured log root.
- Added DGX-to-Selena code synchronization and Selena-to-DGX result/log pulls.
  Selena writes to distinct `outputs_selena/results` and `logs_selena` trees;
  lightweight, detailed, and full pulls reflect TIME's summary, configuration,
  metric, prediction, and scheduler-log artifact sizes.

## Data and evaluation repairs

- Corrected `training_dataset` to exclude validation and test observations and
  `validation_dataset` to end at the test boundary.
- Added combined split-length validation, explicit zero-validation behavior,
  and rejection of non-empty intervals shorter than one forecast horizon.
- Made documentation and configuration consistently state the implemented
  floor-based complete-window rule.
- Rejected forecast arrays whose instance count cannot be reshaped into the
  configured number of windows.
- Removed duplicate Seasonal Naive rows from local leaderboard aggregation and
  made result/cache paths configurable.

## Feature and command repairs

- Removed an unused default tsfeatures list that misleadingly named
  `heterogeneity` even though the runner never computed it.
- Stopped feature-module import from globally suppressing every warning in the
  host process.
- Removed the accidental default `Oil_Price/B` selection so the feature runner
  now requires `--dataset` or `--all` as its control flow and help text claim.
- Documented the actual STL/MSTL output directories, frequency-domain columns,
  per-variate scope, and absence of feature binarization or cross-variate
  heterogeneity.

## Packaging and documentation repairs

- Removed unused `hydra-core`, `ray`, `orjson`, and `matplotlib` dependencies;
  declared packages imported directly by the common code.
- Removed the unused Hatch dynamic-version and nonexistent root `config/`
  source-distribution declarations.
- Replaced placeholder project URLs and aligned package/license metadata with
  the README's Apache-2.0 declaration.
- Fixed the broken `DATA_FORMAT.md` link, its stale link label, obsolete
  `output/features` path, and the claim that the complete evaluation interface
  does not use GluonTS.
- Corrected the prediction archive documentation: `predictions.npz` contains
  quantile predictions and levels, not copied ground truth.
- Added one dependency-light maintenance contract covering Python syntax,
  configuration parsing, local documentation links, private lifecycle ignores,
  and the declared split offsets.
