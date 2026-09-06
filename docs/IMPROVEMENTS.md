# Delta from upstream TIME

Improved currently derives from `zqiao11/TIME` commit
`c11ed82c3eaf39e42e081e5995e7880a76f86cb9`. This page records intentional
reusable divergences so later upstream merges can preserve or retire them
deliberately.

## Runtime infrastructure

- Added one environment-variable path contract and local artifact
  placeholders for datasets, weights, outputs, and logs.
- Added `TIME_METADATA` below the shared dataset root for reusable quality and
  feature artifacts, keeping dataset-derived state independent of project
  result directories while leaving the saved-Arrow tree immutable.
- Added an explicit preparation-host downloader for an immutable revision of
  the official `Real-TSF/TIME` saved-Arrow tree, with ordinary HTTP transfer,
  revision-pinned interrupted-download recovery, and post-download format
  validation.
- Routed maintained runs directly below their launched `foundation_models` or
  `channels_comparison` experiment, with separate `tasks`, `summary`, and
  `feature_analysis` responsibilities instead of generic `results`, `expe_*`,
  `aggregates`, or `analysis` containers. Each task retains canonical model
  alias, actual target mode, identity, monotonic `run_n`, and a plain schema-1
  configuration manifest.
- Added cross-experiment compact-result reuse. A strict explicit source fails
  unless exactly one completed run matches the complete scientific identity;
  an optional source falls back to normal computation. The multivariate
  channel case opportunistically imports matching Chronos-2 `config.json` and
  `metrics_summary.json` artifacts from the foundation-model experiment and
  records their source manifest.
- Added task-boundary recovery before foundation inference. The default exact
  policy skips completed configurations, resumes interrupted configurations in
  their existing `run_n`, records every launch/job/time attempt, and allocates
  new directories only for different configurations or explicit repeats.
  Failed model and comparison jobs mark their still-running task manifests
  interrupted; recovery restarts the current task rather than checkpointing
  inside inference. Resume records identify both the new and preceding jobs.
  Run discovery recognizes only manifests directly below `run_n`, so nested
  component manifests cannot be mistaken for task lifecycle state.
  Reports expose selected, latest, distinct, and averaged repeat handling plus
  error, distinct, latest, and averaged scientific-configuration handling.
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
  priority. Runtime setup creates writable roots without fabricating the
  saved-Arrow leaf, so model and diagnostic jobs reject a missing
  `TIME_DATASET` before entering uv.
- Avoided Hugging Face `datasets` 2.x's NumPy formatter at the saved-Arrow
  boundary. TIME now keeps rows in their native representation and converts
  only numeric GluonTS fields with `np.asarray`, preserving values while
  remaining compatible with the shared NumPy 2 environment.
- Made every retained runner fail on its first unsuccessful dataset instead of
  printing a traceback, continuing the sweep, and returning exit code zero.
  Workflow completion markers and dependent summaries now reflect the runner's
  actual terminal status.
- Completed the previously headerless Seasonal Naive shell runner with
  project-root resolution and the same prepared-environment contract.
- Added one accelerator-synchronized test-loop timer used by every model
  runner. Each task stores total inference seconds in `config.json`, excluding
  model loading, dataset construction, metric computation, and result saving.
- Added a four-model runner and CSV/Markdown summary. Every task MASE is now
  divided by its matching Seasonal Naive MASE and the resulting TIME
  leaderboard ratios are geometrically averaged; timing totals require
  complete task coverage.
- Made the dependent foundation summary generate the launch-filtered
  MASE-versus-feature SVG, joined data, and correlation table after confirming
  that all four model jobs completed successfully. Transfer and publication
  retain these compact analysis artifacts.
- Added a compact `metrics_summary.json` beside every task result, containing
  finite aggregate metrics and coverage counts. Lightweight result transfer
  and publication include these files, while detailed transfer retains raw
  per-window `metrics.npz` arrays.
- Made foundation summaries, the local TIME leaderboard, and feature-versus-
  performance analysis select completed manifests. Exact repeats use an
  automatically advanced or explicitly pinned selection by default; latest,
  distinct, and hierarchical-average policies remain explicit alternatives.
  Different scientific configurations fail closed unless filtered or handled
  with distinct, latest, or hierarchical-average policy. Aggregate report
  manifests list their exact run inputs. Lightweight transfer and publication
  include run and aggregate manifests, repeat selections, and manifest history.
- Added matching DGX and Selena Slurm fronts for each retained model plus a
  separate dependent summary front. One submission helper creates four
  independently schedulable model jobs and starts the summary after all four
  terminate. The report filters task artifacts by launch ID and includes each
  model's terminal status, so a failed model yields an explicit partial report
  rather than blocking aggregation or mixing stale tasks. Dataset/frequency
  and horizon-term loops remain sequential inside each model allocation. Both
  clusters emit explicit task/workflow completion records and durable status
  files below the configured log root.
- Added one channel-comparison submission helper that launches Chronos-2 on
  every multivariate dataset in native multivariate, independent univariate,
  and past-target-covariate modes under one shared launch ID. Each mode remains
  an independently schedulable DGX or Selena job with its own aggregate.
- Made each channel-comparison aggregate record the completed Chronos-2
  evaluation state explicitly. Its workflow status file is keyed by comparison
  mode rather than model alias and is not yet terminal while its in-job summary
  is being generated, so the summary no longer reports a blank state.
- Added DGX-to-Selena code synchronization and Selena-to-DGX result/log pulls.
  Selena writes to the standard `outputs/` and `logs/` trees below its scratch
  project root; DGX pulls them into distinct local `outputs/selena/` and
  `logs/selena/` subtrees. Selena has no current runtime directories named
  `outputs_selena/` or `logs_selena/`; the `selena/` namespace is added only
  on DGX after synchronization. Lightweight, detailed, and full pulls reflect
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
- Made job-specific transfer and publication retain structured workflow status
  records, not only scheduler stdout/stderr pairs.
- Made job-specific transfer and publication retain compact dataset-metadata
  aggregates exported below the project log root, without duplicating exact
  positions or per-variate features in experiment outputs.
- Made dataset-metadata exports stage-specific and atomic per file: successful
  audits export only audit summaries, while the shared feature index enters the
  same job export only after feature extraction succeeds.
- Included the Slurm job and array identifiers in the first workflow log line
  as well as the durable status file, making a copied stdout/stderr pair
  self-identifying after it leaves the scheduler.
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
- Normalized both TS-ICL forecast return forms: one tensor for stackable
  contexts and a list of tensors for variable-length contexts. Both now enter
  TIME evaluation as `(batch, quantile, variate, horizon)` arrays.
- Declared TS-ICL's target-channel behavior accurately: upstream moves multiple
  targets into its batch dimension, so target-only evaluation is univariate
  and parallelized rather than native multivariate. Explicit covariates still
  enter TS-ICL's channel mixer; unsupported multivariate requests fail.
- Added explicit foundation covariate capability checks. Chronos-2 and TS-ICL
  accept known `L+H` dataset covariates; Chronos-2 also forecasts each target
  separately from its `L` history with every other target history as past-only
  covariates. Unsupported modes and absent or structurally invalid covariates
  fail instead of being ignored.
- Aligned capable-backbone covariates with target-context missingness: any
  non-finite observation is represented as NaN for the backbone's ordinary
  missing-value mask instead of aborting before inference.
- Suppressed only the known pandas frequency-alias deprecation messages that
  Seasonal Naive amplified once per forecast window, while retaining unrelated
  warnings. Undefined all-zero MAPE/sMAPE cells now remain NaN without emitting
  NumPy empty-mean warnings.
- Removed stochastic inverse-CDF resampling from Seasonal Naive. Its
  StatsForecast quantiles now enter TIME evaluation directly, making repeated
  runs deterministic without inventing a second random-number stream.
- Corrected MASE scaling so genuine missing timestamps are never removed from
  the calendar. Only finite seasonal pairs contribute, and their exact count
  is the denominator used by the seasonal-error mean.
- Made foundation, channel, and feature-performance comparisons use task MASE
  divided by the matching corrected Seasonal Naive MASE. Channel summaries
  consequently require the completed foundation Seasonal Naive tasks.
- Removed duplicate Seasonal Naive rows from local leaderboard aggregation and
  made result/cache paths configurable.

## Feature and command repairs

- Removed an unused default tsfeatures list that misleadingly named
  `heterogeneity` even though the runner never computed it.
- Stopped feature-module import from globally suppressing every warning in the
  host process.
- Removed the accidental default `Oil_Price/B` selection so the feature runner
  now requires `--dataset` or `--all` as its control flow and help text claim.
- Added saved-Arrow feature input, per-dataset summaries, temporal and spatial
  location/scale/frequency heterogeneity, dataset ranks, and top-correlated
  feature-versus-MASE SVG/CSV analysis. Documented the actual STL/MSTL output
  directories, frequency-domain columns, per-variate scope, and absence of
  feature binarization.
- Added a shared source/window audit that records non-finite source positions
  once, evaluates each distinct context-limit/forecast-horizon pair once via
  prefix counts, detects constant windows from adjacent-value transitions,
  and maps model profiles to reusable window configurations. Full-series
  feature extraction shares the same metadata root and reuses complete files.
- Repaired `seasonal_corr` by excluding unusable cycles, preserving variates
  for which that optional correlation is undefined, and replacing quadratic
  pair enumeration with an equivalent linear-time mean. Feature reuse now
  validates source variate identities and computes only rows missing from a
  partial artifact.

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
