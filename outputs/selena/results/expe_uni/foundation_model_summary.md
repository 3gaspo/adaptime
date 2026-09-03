# Foundation-model benchmark summary

MASE is averaged equally over available H settings within each dataset/frequency, then equally over dataset/frequency entries. Inference seconds are summed over the same test forecasting tasks; a blank total means at least one task lacks timing metadata.

| Model | Target mode | State | Exit | MASE (macro) | Inference seconds | Datasets | Tasks | Timed tasks |
|---|---|---|---:|---:|---:|---:|---:|---:|
| chronos2 | multivariate,univariate | completed | 0 | 1.122040 | 431.509 | 50 | 98 | 98 |
| ts_icl | univariate | completed | 0 | 1.167467 | 2835.570 | 50 | 98 | 98 |
| chronos_bolt | univariate | completed | 0 | 1.238911 | 948.589 | 50 | 98 | 98 |
| seasonal_naive | univariate | completed | 0 | 1.553036 | 2316.931 | 50 | 98 | 98 |
