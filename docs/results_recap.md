# Completed experiment recap

## Current 90-task evidence

The September 2026 rerun applies the shared Seasonal-defined evaluation grid,
whole-cell vanilla fallback for non-finite method predictions, and the standard
four-dataset/frequency exclusion list. The resulting scope is 90 TIME tasks.
Shared preparation, canonical Chronos-2 vanilla forecasting, Ridge extraction,
and the independent TS-RAG extraction, prediction, and evaluation phases are
complete for all 90 tasks.

The Ridge branch stopped during fitting because the execution host exhausted
its scratch quota. Thirty task fits completed and the next fit is incomplete;
no Ridge-family test prediction or evaluation was produced. Consequently, the
intended Adaptime-versus-TS-RAG comparison and unified report are not yet
available.

## Independent TS-RAG result

TS-RAG has complete current-contract MASE coverage on all 90 tasks. Relative
to the matched Seasonal Naive result, its task-level MASE is lower on 83 tasks
and higher on seven. The arithmetic mean of task MASE values is 1.1376 for
TS-RAG versus 1.4454 for Seasonal Naive, and the median within-task relative
change is -24.2%.

| Term | Tasks | TS-RAG wins | Mean TS-RAG MASE | Mean Seasonal MASE | Median relative change |
|---|---:|---:|---:|---:|---:|
| Short | 46 | 44 | 1.091 | 1.498 | -31.3% |
| Medium | 22 | 20 | 1.060 | 1.289 | -20.7% |
| Long | 22 | 19 | 1.313 | 1.491 | -17.1% |

The new fallback contract repaired all 16 task configurations that previously
lost finite MASE coverage under TS-RAG. MSE, MAE, RMSE, sMAPE, and MASE are
finite on the complete selected grid for every task. The remaining MAPE, ND,
and CRPS gaps occur only on `SG_Carpark/15T/{short,medium}` and exactly match
the Seasonal Naive finite support; TS-RAG has additional finite sMAPE values on
those two tasks. The synchronized lightweight summaries do not include the
fallback masks, so the number of cells replaced by vanilla cannot yet be
reported.

This is useful control evidence, but it does not answer the research question:
TS-RAG has not yet been compared with canonical vanilla or the validation-
selected Adaptime family on completed common-support test evaluations.

## Preliminary selector behavior

The 30 completed Ridge fits cover an execution-order prefix drawn from
`Water_Quality_Darwin`, `current_velocity`, `CPHL`, and `Coastal_T_S`. The
overall validation selector retained a non-vanilla candidate on 23 tasks:
`bayes_past_targets_prediction` on 14, `bayes_covariate_prediction` on five,
`full_ridge_per_variate` on three, and `full_ridge_shared` on one. It selected
vanilla on seven.

Across the 30 fits, the family-specific selectors retained a non-vanilla
candidate on 14 tasks for the past-target Bayesian family, 12 for the
retrieval-covariate Bayesian family, eight for `y_ridge_shared`, seven each for
`cov_ridge_shared` and `full_ridge_shared`, and four for
`full_ridge_per_variate`.

These counts describe validation selection only. They are execution-order
biased, omit two thirds of the task grid, and contain no official-test metrics;
they therefore support no claim about generalization or proposal superiority.

## Required evidence

The execution host must regain enough scratch quota to resume the incomplete
Ridge fit while preserving the completed 90 extractions and 30 fits. The Ridge
branch must then finish fitting, selected-K test extraction, prediction, and
evaluation before the unified report can compare Adaptime, vanilla, TS-RAG,
and Seasonal Naive on identical support. Rolling Ridge and independent
inference timing remain separate, unrun experiments.
