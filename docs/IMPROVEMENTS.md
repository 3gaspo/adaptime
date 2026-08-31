# Delta from upstream TIME

Improved currently derives from `zqiao11/TIME` commit
`c11ed82c3eaf39e42e081e5995e7880a76f86cb9`. This page records intentional
reusable divergences so later upstream merges can preserve or retire them
deliberately.

## Runtime infrastructure

- Added one environment-variable path contract and local artifact
  placeholders for datasets, weights, outputs, and logs.
- Added an explicit preparation-host downloader for an immutable revision of
  the official `Real-TSF/TIME` saved-Arrow tree, with an empty-destination
  requirement and post-download format validation.
- Routed every retained model runner's default results below
  `TIME_OUTPUTS/results` using its canonical model alias.
- Narrowed the maintained benchmark surface to `chronos_bolt`, `chronos2`,
  `ts_icl`, and `seasonal_naive`. Removed every other model adapter and runner,
  including Toto, TiRex, and TimesFM 2, plus the unused non-seasonal Naive
  wrapper. Toto was removed because its exact NumPy and scikit-learn pins
  conflict with TS-ICL in the required single shared environment; TiRex is
  deferred until its checkpoint is prepared. Current summaries ignore old
  result directories outside this set.
- Added shared Hugging Face/Torch cache defaults and a cluster wrapper that
  calls the existing TIME run scripts through a prepared uv
  environment with externally configured paths. Removed the runners' Conda
  and job-time package installation; the Slurm jobs never mutate their
  environments themselves.
- Aligned learned-model loading with the TSFM offline contract. Chronos-2,
  Chronos-Bolt, and TS-ICL resolve explicit local checkpoints below
  `TIME_WEIGHTS`; Chronos uses local-only loading, TS-ICL disables automatic
  download, and Selena exports the upstream offline-mode switches.
- Made Selena load its preserved project `.env` before applying cluster path
  defaults while retaining submission-time environment overrides as highest
  priority. Model jobs reject a missing `TIME_DATASET` before entering uv.
- Made every retained runner fail on its first unsuccessful dataset instead of
  printing a traceback, continuing the sweep, and returning exit code zero.
  Workflow completion markers and dependent summaries now reflect the runner's
  actual terminal status.
- Completed the previously headerless Seasonal Naive shell runner with
  project-root resolution and the same prepared-environment contract.
- Added one accelerator-synchronized test-loop timer used by every model
  runner. Each task stores total inference seconds in `config.json`, excluding
  model loading, dataset construction, metric computation, and result saving.
- Added a four-model runner and CSV/Markdown summary whose MASE
  macro-average weights H settings equally within dataset/frequency entries
  and then weights those entries equally; timing totals require complete task
  coverage.
- Added matching DGX and Selena Slurm fronts for each retained model plus a
  separate dependent summary front. One submission helper creates four
  independently schedulable model jobs and starts the summary only after all
  four succeed. Dataset/frequency and horizon-term loops remain sequential
  inside each model allocation. Both clusters emit explicit task/workflow
  completion records and durable status files below the configured log root.
- Added DGX-to-Selena code synchronization and Selena-to-DGX result/log pulls.
  Selena writes to the standard `outputs/` and `logs/` trees below its scratch
  project root; DGX pulls them into distinct local `outputs/selena/` and
  `logs/selena/` subtrees. Lightweight, detailed, and full pulls reflect
  TIME's summary, configuration, metric, prediction, and scheduler-log
  artifact sizes. Code synchronization excludes every output and log tree.
- Kept DGX and Selena uv environments independent while making every retained
  model on a given cluster use that cluster's single shared environment. Code
  synchronization preserves Selena's `.venv`, `pyproject.toml`, and `uv.lock`,
  including when `.venv` is a symlink. Selena loads `python/3.12_pypsa` before
  uv runs and forbids uv-managed Python downloads. Dataset and weight payloads
  now live outside the code tree, so code synchronization no longer needs
  exclusions for those project-relative names or removed external source
  checkouts.
- Added a manual DGX publisher that selects both native and synchronized
  Selena logs and outputs. Its TIME-specific lightweight, detailed, and full
  tiers mirror result synchronization, replace oversized files with bounded
  metadata/text samples, pull `origin/main` through the configured proxy, and
  push only the selected artifact paths plus existing local commits.
- Added a shared storage-root override so DGX resolves datasets and weights
  below the user home outside `codes/`, while Selena resolves them beside
  `codes/` in user scratch. Project outputs and logs remain project-owned.
- Exported each Slurm front's `PROJECT_ROOT` so individual model and summary
  child Bash processes preserve the submitted project checkout.

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
