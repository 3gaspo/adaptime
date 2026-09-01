# Foundation-model benchmark summary

MASE is averaged equally over available H settings within each dataset/frequency, then equally over dataset/frequency entries. Inference seconds are summed over the same test forecasting tasks; a blank total means at least one task lacks timing metadata.

| Model | State | Exit | MASE (macro) | Inference seconds | Datasets | Tasks | Timed tasks |
|---|---|---:|---:|---:|---:|---:|---:|
| chronos2 | completed | 0 | 1.122040 | 436.629 | 50 | 98 | 98 |
| ts_icl | completed | 0 | 1.167895 | 2553.966 | 50 | 98 | 98 |
| chronos_bolt | completed | 0 | 1.238911 | 954.222 | 50 | 98 | 98 |
| seasonal_naive | completed | 0 | 1.551718 | 2313.758 | 50 | 98 | 98 |
