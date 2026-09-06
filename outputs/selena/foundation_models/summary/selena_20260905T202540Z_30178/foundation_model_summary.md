# Foundation-model benchmark summary

MASE is averaged equally over available H settings within each dataset/frequency, then equally over dataset/frequency entries. Inference seconds are summed over the same test forecasting tasks; a blank total means at least one task lacks timing metadata.

| Model | Target mode | State | Exit | MASE (macro) | Inference seconds | Datasets | Tasks | Timed tasks |
|---|---|---|---:|---:|---:|---:|---:|---:|
| chronos2 | multivariate,univariate | completed | 0 | 1.122040 | 429.442 | 50 | 98 | 98 |
| ts_icl | univariate | completed | 0 | 1.167467 | 2822.646 | 50 | 98 | 98 |
| chronos_bolt | univariate | completed | 0 | 1.238911 | 942.774 | 50 | 98 | 98 |
| seasonal_naive | univariate | completed | 0 | 1.555939 | 2313.715 | 50 | 98 | 98 |
