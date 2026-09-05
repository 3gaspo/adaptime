# Foundation-model results

## Evidence scope

Selena launch `selena_20260902T183319Z_29483` completed the maintained
four-model TIME benchmark on 2026-09-02. Chronos2, TS-ICL, Chronos-Bolt, and
Seasonal Naive each completed the same 98 horizon tasks spanning 50
dataset/frequency entries. The four model jobs and dependent summary exited
zero. All 392 selected task manifests use schema 1 and report `completed`.

Channel launch `selena_channels_20260902T183320Z_29592` separately evaluated
Chronos2 on the 74 tasks from the 38 multivariate dataset/frequency entries.
Its native-multivariate and independent-univariate jobs both completed all 74
tasks. After the missing-covariate repair, launch
`selena_covariate_3120322` reran the complete 74-task past-target branch on
2026-09-04 and completed with exit code zero.

Dataset-diagnostics launch
`selena_dataset_diagnostics_20260904T155618Z_29769` also completed on
2026-09-04. It audited the 50 configured dataset/frequency entries and exported
490 distinct query/context configurations plus 50 complete dataset-level
feature rows.

The synchronized project-local copies of these logs and result artifacts were
intentionally removed on 2026-09-05 before a clean rerun under the current
task-recovery and experiment-path contract. The values below remain the
historical analysis of the named completed launches; they are not current
selectable inputs in this checkout. Shared dataset data and computed metadata
were not removed.

The analysis uses MASE because it is scale-free and remains interpretable on
the near-zero series for which MAPE becomes unstable. For each model or target
mode, macro MASE first averages available horizons within a dataset/frequency
entry and then weights dataset/frequency entries equally. Summed inference
time covers the forecast loops only and excludes model loading, dataset
construction, metric computation, and artifact saving.

## Main result

| Model | Macro MASE ↓ | Inference seconds ↓ | Best of 98 tasks | Best of 50 datasets |
|---|---:|---:|---:|---:|
| Chronos2 | **1.122040** | **431.509** | **74** | **38** |
| TS-ICL | 1.167467 | 2835.570 | 17 | 9 |
| Chronos-Bolt | 1.238911 | 948.589 | 6 | 2 |
| Seasonal Naive | 1.553036 | 2316.931 | 1 | 1 |

Chronos2 is the clear default for this benchmark: it has both the lowest macro
MASE and the lowest measured forecast-loop time, so it dominates the other
three models on the observed accuracy/runtime plane. It beats TS-ICL on 78 of
98 tasks, Chronos-Bolt on 88, and Seasonal Naive on 97. Relative to Seasonal
Naive, its median dataset-level MASE ratio is 0.664, corresponding to a 33.6%
median reduction, and it improves on the baseline on 49 of 50 datasets.

TS-ICL provides the second-best macro MASE and wins 17 tasks and 9 datasets,
but its measured inference total is 6.57 times Chronos2's and 2.99 times
Chronos-Bolt's. Its dataset wins are `Australia_Solar/H`, `CPHL/H`,
`Commodity_Import/M`, `EWELD_Load/15T`, `Housing_Inventory/M`,
`NE_China_Wind/H`, `US_Labor/M`, `Uncertainty_1M/M`, and
`epf_electricity_price/H`.

Chronos-Bolt is the second-fastest learned model and wins `ECDC_COVID/W` and
`Vehicle_Sales/M`, but its aggregate accuracy is below both Chronos2 and
TS-ICL. Its upstream warning on the 25 tasks with prediction length above 64
is a model applicability limitation: those forecasts completed, but the model
is not optimized for those horizons.

Seasonal Naive is useful as a scale-aware baseline rather than a competitive
default. It wins only `Job_Claims/W`; that is also the only dataset where
Chronos2 does not improve on it. Its high measured runtime reflects the
current per-window StatsForecast evaluation wrapper and should not be
interpreted as the intrinsic cost of copying a seasonal lag.

## Horizon behavior

| Horizon subset | Tasks | Chronos2 | TS-ICL | Chronos-Bolt | Seasonal Naive |
|---|---:|---:|---:|---:|---:|
| Short | 50 | **1.0143** | 1.0541 | 1.1044 | 1.4726 |
| Medium | 24 | **0.9148** | 0.9530 | 1.0361 | 1.2652 |
| Long | 24 | **1.1152** | 1.1649 | 1.2610 | 1.4819 |

Chronos2 leads every horizon subset. All models are strongest on the medium
subset and degrade on the long subset, while the short subset covers more and
different datasets than medium and long. These rows are therefore descriptive
subset means, not a controlled horizon-only causal comparison.

## Chronos2 channel comparison

| Target representation | Macro MASE ↓ | Inference seconds ↓ | Datasets | Tasks |
|---|---:|---:|---:|---:|
| Native multivariate | **1.094275** | **328.077** | 38 | 74 |
| Independent univariate | 1.111994 | 351.322 | 38 | 74 |
| Past targets as covariates | 1.094275 | 1908.327 | 38 | 74 |

On the paired multivariate subset, native multivariate inference reduces
macro MASE by 1.59% and measured inference time by 6.62%. It wins 46 of 74
tasks and 24 of 38 dataset/frequency aggregates. The effect is heterogeneous:
the largest dataset-level gains occur on `Crypto/D`, `Global_Price/Q`,
`JOLTS/M`, `Vehicle_Supply/M`, and `Coastal_T_S/20T`, while the largest loss is
on `US_Labor/M`. Native multivariate inference is therefore the stronger
default across this subset, but it is not uniformly superior.

The completed past-target branch has macro MASE `1.094275206`, numerically
indistinguishable at six decimals from native multivariate's `1.094275190`.
It is 5.82 times slower than native multivariate and 5.43 times slower than
independent univariate under the measured adapters. This total is valid for
the complete rerun: launch 3120322 forecast all 74 tasks exactly once, including
new `run_1` artifacts for the six tasks produced by the earlier failed launch.
The earlier six task timings are not added to the 1908.327-second total.

## Missing-value and constant-window diagnostics

The original past-target job `3030434` failed because raw target histories
reused as covariates contained missing observations. The repaired adapter now
represents every non-finite covariate observation as NaN and lets the capable
backbone apply the same missing-value mask used for target contexts. The
completed rerun confirms that this policy restores all 74 tasks.

The reusable audit evaluated 202,935 generated queries and 555,885 channel
windows over the distinct maintained `(L,H)` configurations. It found:

- 22,066,434 non-finite values across 136,029 channel windows and 19
  dataset/frequency entries;
- every non-finite value was NaN, with no positive or negative infinity;
- 693 finite constant query windows across seven dataset/frequency entries;
- 106 all-NaN query windows across four dataset/frequency entries.

These are counts over reusable window configurations, not unique source-cell
counts: a source observation can participate in multiple contexts and
horizons. Exact source positions are deduplicated in shared dataset metadata.
Constant and all-NaN output windows can make a scale-free metric undefined;
the benchmark preserves those cell-level NaNs and reports finite aggregate
coverage rather than imputing ground truth.

The feature repair produced all 50 dataset rows. `seasonal_corr` is no longer
lost merely because one seasonal cycle is unusable. The only blank aggregate
feature fields are optional second/third detected periods and strengths for 11
datasets, which means those additional periods were not identified rather than
that feature extraction failed.

## Integrity and limitations

- Every main-run model has 98 completed task manifests, timing for all 98
  tasks, and a defined task-level MASE aggregate. Cell-level MASE coverage is
  identical across models for every task, so there is no model-specific
  coverage loss.
- Seasonal Naive and the dependent summary have empty stderr. The completed
  learned-model and channel logs contain only dependency deprecations,
  serialization notices, model-loading progress, and Chronos-Bolt's documented
  horizon warning; they contain no traceback, CUDA error, or out-of-memory
  marker.
- The results describe one completed execution of each fixed pretrained model,
  not repeated stochastic trials; no uncertainty interval or significance
  claim is available.
- Inference totals measure the present TIME adapters and batching choices.
  They support operational comparison of these implementations, not a general
  hardware-independent claim about backbone complexity.
- Lightweight artifacts support manifest, aggregate, task-level, and log
  analysis. Raw per-window metrics require detailed synchronization, and
  prediction arrays require the full tier.
- The past-target aggregate's CSV has blank lifecycle fields because its
  summary was generated before the comparison-status reporting repair. Its
  authoritative workflow status is `completed` with exit code zero; future
  comparison summaries receive that successful evaluation status explicitly.
