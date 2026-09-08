# Completed experiment recap

## Current partial Adaptime evidence

The current synchronized run attempted the 98-task, Chronos-2 univariate
`full_ridge_shared` workflow under the window-proportional, 8192-context,
MSSE-fitting, and Seasonal-Naive-scaled-MASE contract. It allocated 62 tasks
before failing: 61 completed, one was interrupted, and the remaining 36 were
never allocated.

Of the 61 completed tasks, 44 correctly exercised the configured
vanilla-only fallback because their histories could not supply the requested
chronological adaptation regions and context. Seventeen tasks fitted and
evaluated a ridge adaptor. Across only those 17 fitted-ridge tasks:

| Method | Scaled MASE (geometric mean) | Test-time seconds |
|---|---:|---:|
| Seasonal Naive | 1.000000 | -- |
| Vanilla Chronos-2 | 0.758386 | 15.566 |
| Retrieval-covariate Chronos-2 | 0.775540 | 1225.309 |
| Adaptime `full_ridge_shared` | 0.775212 | 1241.391 |

Adaptime is 2.22% worse than vanilla in geometric mean on this partial fitted
subset. It wins 7 of 17 task aggregates, with a median Adaptime/vanilla ratio
of 1.0044, a best ratio of 0.9812, and a worst ratio of 1.2706. The raw
retrieval-covariate branch is 2.26% worse than vanilla and wins one task.
Adaptime takes 0.369 seconds per test window versus 0.00463 for vanilla, about
79.8 times longer, before separately accounting for the precomputed extraction
stage.

Retrieval is eligible for every test window on 15 fitted tasks and one third
of test windows on each of the two Australia Solar tasks, giving an unweighted
mean task eligibility of 92.2%. Validation selected `K=1` for 9 tasks, `K=10`
for 3, and `K=15` for 5; it never selected `K=5`. It selected alpha `1e-3` for
6 tasks, `1e-2` for 4, and `1e-1` for 7. This spread does not indicate one
stable global ridge configuration.

Including the 44 vanilla-only fallbacks, the incomplete 61-task prefix scores
0.707292 for vanilla, 0.711715 for the covariate branch, and 0.711631 for
Adaptime. Those fallback tasks make all three methods identical and therefore
do not provide adaptation evidence. The fitted subset is also execution-order
selected rather than a complete TIME sample. The current partial evidence does
not support the hypothesis that `full_ridge_shared` improves vanilla
Chronos-2, but it is not a final benchmark conclusion.

## Failure diagnosis

The interrupted task is `SG_Carpark/15T/medium`. Its prepared manifest contains
4,956 adaptation-training, 2,478 validation, 2,478 test, and 217,002 datastore
windows. The extraction manifest rejects every datastore window for a
non-finite future target, leaving zero eligible train, validation, or test
retrieval rows. Ridge fitting therefore receives empty sufficient statistics
and raises `cannot solve empty ridge statistics`. The lightweight publication
does not include the source arrays needed to identify the exact non-finite
dates or channels.

The current protocol classifies this case as a vanilla-only fallback whenever
the primary `K=10` has no more valid training dates than official test dates.
The repair changes the scientific identity, so the old 61-task prefix is
evidence for the superseded run only; the current Ridge workflow requires a
fresh evaluation.

## TS-RAG status

The matched TS-RAG run produced no result. It started while the Adaptime run
was still running and immediately required a selected completed ridge for
`Water_Quality_Darwin/15T/short`, which was instead a valid vanilla-only
fallback. The replacement workflow makes TS-RAG independent of Ridge and
evaluates both wrappers through the same TIME evaluator. A complete rerun is
the next evidence needed.
