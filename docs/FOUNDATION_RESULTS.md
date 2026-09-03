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
tasks. The past-target-covariate job failed after completing only six tasks, so
that branch is diagnostic evidence rather than a complete channel comparison.

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

On the paired multivariate subset, native multivariate inference reduces
macro MASE by 1.59% and measured inference time by 6.62%. It wins 46 of 74
tasks and 24 of 38 dataset/frequency aggregates. The effect is heterogeneous:
the largest dataset-level gains occur on `Crypto/D`, `Global_Price/Q`,
`JOLTS/M`, `Vehicle_Supply/M`, and `Coastal_T_S/20T`, while the largest loss is
on `US_Labor/M`. Native multivariate inference is therefore the stronger
default across this subset, but it is not uniformly superior.

The incomplete past-target branch finished all three horizons for only
`Water_Quality_Darwin/15T` and `current_velocity/5T`. Across those six tasks,
its two-dataset macro MASE is 0.770388 versus 0.770556 for independent
univariate inference: effectively neutral and far too narrow for a general
conclusion.

## Past-target failure diagnosis

The past-target job `3030434` failed on the first short-horizon batch of
`current_velocity/10T` with `ValueError: Covariates must be finite over the
complete L window`. The execution path is:

1. `MultivariateToUnivariateWithPastTargets` selects one target variate and
   copies the other five raw target histories into `past_feat_dynamic_real`.
2. `extract_covariate_window` requires every copied value in the historical
   covariate block to be finite.
3. At least one copied `current_velocity/10T` value is non-finite, so the
   strict covariate check aborts before Chronos2 prediction.

This is a data-handling contract gap specific to reusing targets as
covariates. Ordinary target evaluation tolerates missing historical target
values and completed all `current_velocity/10T` tasks, while the new
past-target path forwards those values unchanged into a stricter interface.
The lightweight log reports neither item ID, target/covariate channel, window
bounds, nor non-finite count, so the exact source cell cannot be identified
from the synchronized artifacts. A corrective rerun first needs an explicit
scientific policy for missing past-target covariates, such as imputation or
task exclusion, plus coordinate-rich failure diagnostics.

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
