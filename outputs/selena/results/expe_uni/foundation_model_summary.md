# Foundation-model benchmark summary

MASE is averaged equally over available H settings within each dataset/frequency, then equally over dataset/frequency entries. Inference seconds are summed over the same test forecasting tasks; a blank total means at least one task lacks timing metadata.

| Model | Target mode | State | Exit | MASE (macro) | Inference seconds | Datasets | Tasks | Timed tasks |
|---|---|---|---:|---:|---:|---:|---:|---:|
| chronos2 | multivariate,univariate | completed | 0 | 1.122040 | 442.119 | 50 | 98 | 98 |
| ts_icl | univariate | completed | 0 | 1.167467 | 2786.841 | 50 | 98 | 98 |
| chronos_bolt | univariate | completed | 0 | 1.238911 | 943.949 | 50 | 98 | 98 |
| seasonal_naive | univariate | completed | 0 | 1.557194 | 2299.613 | 50 | 98 | 98 |
