# Foundation-model results

## Evidence scope

Selena launch `selena_20260901T173551Z_23108` completed the maintained
four-model TIME benchmark on 2026-09-01. Chronos2, TS-ICL, Chronos-Bolt, and
Seasonal Naive each completed the same 98 horizon tasks spanning 50
dataset/frequency entries. Every model and the dependent summary exited zero;
the synchronized lightweight artifacts contain launch-stamped task
configuration, compact finite-metric summaries, workflow status, and the final
aggregate table.

This analyzed launch predates the current per-run manifest layout and explicit
channel-comparison workflow. It remains historical evidence for that executed
adapter configuration; current result readers do not reuse it as a completed
manifested run, and the updated contract requires a full four-model rerun.

The analysis uses MASE because it is scale-free and remains interpretable on
the near-zero series for which MAPE becomes unstable. For each model, the
reported macro MASE first averages available horizons within a
dataset/frequency entry and then weights the 50 entries equally. Summed
inference time covers the forecast loops only and excludes model loading,
dataset construction, metric computation, and artifact saving.

## Main result

| Model | Macro MASE ↓ | Inference seconds ↓ | Best of 98 tasks | Best of 50 datasets |
|---|---:|---:|---:|---:|
| Chronos2 | **1.122040** | **436.629** | **73** | **37** |
| TS-ICL | 1.167895 | 2553.966 | 18 | 10 |
| Chronos-Bolt | 1.238911 | 954.222 | 6 | 2 |
| Seasonal Naive | 1.551718 | 2313.758 | 1 | 1 |

Chronos2 is the clear default for this benchmark: it has both the lowest macro
MASE and the lowest measured forecast-loop time, so it dominates the other
three models on the observed accuracy/runtime plane. It beats TS-ICL on 77 of
98 tasks, Chronos-Bolt on 88, and Seasonal Naive on 97. Relative to Seasonal
Naive, its median dataset-level MASE ratio is 0.673, corresponding to a 32.7%
median reduction, and it improves on the baseline on 49 of 50 datasets.

TS-ICL provides the second-best macro MASE and wins 18 tasks and 10 datasets,
but its measured inference total is 5.85 times Chronos2's and 2.68 times
Chronos-Bolt's. Its dataset wins are `Australia_Solar/H`, `CPHL/H`,
`Commodity_Import/M`, `EWELD_Load/15T`, `Housing_Inventory/M`,
`NE_China_Wind/H`, `Oil_Price/B`, `US_Labor/M`, `Uncertainty_1M/M`, and
`epf_electricity_price/H`.

Chronos-Bolt is the second-fastest learned model and wins `ECDC_COVID/W` and
`Vehicle_Sales/M`, but its aggregate accuracy is below both Chronos2 and
TS-ICL. Its upstream warning on the 25 tasks with prediction length above 64
is a model applicability limitation: those forecasts completed, but the model
is not optimized for those horizons.

Seasonal Naive is useful as a scale-aware baseline rather than a competitive
default. It wins only `Job_Claims/W`; that is also the only dataset where
Chronos2 does not improve on it. Its high measured runtime reflects the current
per-window StatsForecast evaluation wrapper and should not be interpreted as
the intrinsic cost of copying a seasonal lag.

## Horizon behavior

| Horizon subset | Tasks | Chronos2 | TS-ICL | Chronos-Bolt | Seasonal Naive |
|---|---:|---:|---:|---:|---:|
| Short | 50 | **1.0143** | 1.0545 | 1.1044 | 1.4711 |
| Medium | 24 | **0.9148** | 0.9530 | 1.0361 | 1.2617 |
| Long | 24 | **1.1152** | 1.1648 | 1.2610 | 1.4834 |

Chronos2 leads every horizon subset. All models are strongest on the medium
subset and degrade on the long subset, while the short subset covers more and
different datasets than medium and long. These rows are therefore descriptive
subset means, not a controlled horizon-only causal comparison.

## Integrity and limitations

- Every model has 98 task summaries and all task-level MASE aggregates are
  defined. Cell-level missing metric values follow the same ground-truth mask
  across models, so there is no model-specific coverage loss.
- The repaired TS-ICL variable-length output path completed the previously
  failing task and the rest of the sweep. Seasonal Naive produced empty
  stderr after narrow warning suppression. No traceback, exception, runtime
  warning, CUDA error, or out-of-memory marker remains.
- The results describe one completed execution of each fixed pretrained model,
  not repeated stochastic trials; no uncertainty interval or significance
  claim is available.
- Inference totals measure the present TIME adapters and batching choices.
  They support operational comparison of these implementations, not a general
  hardware-independent claim about backbone complexity.
- Channel semantics differed across the completed adapters: Chronos2 performed
  native multivariate inference on multichannel datasets, while TS-ICL,
  Chronos-Bolt, and Seasonal Naive forecast target channels independently.
  TS-ICL's upstream adapter parallelized those target channels in its batch;
  no other target channels were supplied as covariates in this launch.
- Lightweight artifacts support aggregate and task-level analysis. Raw
  per-window metrics require detailed synchronization, and prediction arrays
  require the full tier.
