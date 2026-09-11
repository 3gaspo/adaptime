# Completed experiment recap

## Current partial Adaptime evidence

The 2026-09-09 Selena chain ran the 98-task, Chronos-2 univariate Ridge family
under the window-proportional, 8192-context, MSSE-fitting, and deterministic
Seasonal-Naive-scaled-MASE contract. Preparation, all 98 selected-K evaluation
extractions, and all 98 four-method prediction bundles completed. Common TIME
evaluation then completed the same 18-task execution-order prefix for
`vanilla`, `covariate_prediction`, `bayes_covariate_prediction`, and
`full_ridge_shared` before failing on the nineteenth task.

The 18 completed tasks cover all three `Water_Quality_Darwin` terms and all 15
`current_velocity` frequency/term combinations. Six fitted a Ridge adaptor and
12 used the documented vanilla fallback. Across all 18 aligned tasks:

| Method | Scaled MASE (geometric mean) | Ratio to vanilla | Wins/ties vs vanilla | Test-time seconds |
|---|---:|---:|---:|---:|
| Vanilla Chronos-2 | 0.749478 | 1.000000 | -- | 170.372 |
| Retrieval covariate | 0.753696 | 1.005628 | 2 / 12 | 323.124 |
| Bayesian covariate | 0.747726 | 0.997662 | 5 / 12 | 323.124 |
| Adaptime `full_ridge_shared` | 0.760756 | 1.015048 | 3 / 12 | 341.173 |

Fallback tasks make all four methods identical and do not provide adaptation
evidence. Across only the six fitted tasks:

| Method | Scaled MASE (geometric mean) | Ratio to vanilla | Wins vs vanilla | Test-time seconds |
|---|---:|---:|---:|---:|
| Vanilla Chronos-2 | 0.848132 | 1.000000 | -- | 9.939 |
| Retrieval covariate | 0.862532 | 1.016979 | 2/6 | 162.691 |
| Bayesian covariate | 0.842196 | 0.993002 | 5/6 | 162.692 |
| Adaptime `full_ridge_shared` | 0.886998 | 1.045826 | 3/6 | 180.740 |

Bayesian covariate prediction is 0.70% better than vanilla on this fitted
prefix and wins five of six tasks. Raw covariate prediction is 1.70% worse;
full Ridge is 4.58% worse and reaches its worst ratio, 1.2706, on
`current_velocity/5T/medium`. All 45,024 MASE values are finite for every
method. Because this is a small execution-order-selected prefix rather than a
complete TIME sample, it is diagnostic evidence only: it suggests the Bayesian
mixture is more stable than full Ridge, but supports no final benchmark claim.

## Failure diagnosis

Pipeline job `3241621` failed during the vanilla evaluation of
`CPHL/15T/short`. CPHL has target dimension one and two already-univariate
series. The common `evaluate_point_predictions` path nevertheless constructs
the dataset with unconditional multivariate-to-univariate conversion. That
conversion iterates the one-dimensional target and turns each time series into
scalar targets; GluonTS then accesses `target.shape[-1]` and raises
`IndexError: tuple index out of range`.

This is a common evaluator shape bug, not a Ridge, extraction, prediction, or
saving-path failure. It may affect later datasets that are also natively
univariate. Evaluation should apply multivariate-to-univariate expansion only
when the source target dimension exceeds one. After that repair, the workflow
can reuse all completed preparation, fit extraction, frozen adaptation,
selected-K extraction, and prediction artifacts, resume the interrupted and
remaining 79 evaluation task groups, and then run the dependency-held report.

## Saving-path and evidence boundary

The submitted run used the deployed layout, including vanilla artifacts under
`vanilla/shared/...` and Ridge prediction bundles under `predictions/ridge`.
The later local saving-path rewrite changes where future vanilla artifacts are
written but does not move or scientifically invalidate the submitted results.
Automatic reuse across the old and new vanilla roots must nevertheless be
handled explicitly; payloads should not be copied or merged between layouts.

The synchronized lightweight publication contains manifests, compact task
summaries, workflow records, and logs, but not the heavy prediction arrays.
The partial metrics above are verified at task-summary level. Report job
`3241622` did not run because `afterok:3241621` was not satisfied. TS-RAG was a
separate workflow and was not part of these submitted jobs.
