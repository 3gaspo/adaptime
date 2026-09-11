# Dataset split contract

For a series of length `N`, let:

```text
validation_start = N - test_length - val_length
test_start       = N - test_length
```

The four public views have distinct meanings:

| Property | Content |
|---|---|
| `training_dataset` | Prefix ending at `validation_start`; it excludes both validation and test observations. |
| `validation_dataset` | Prefix ending at `test_start`; it contains training plus validation history and is the information available at the test boundary. The upstream name is retained for API compatibility. |
| `val_data` | GluonTS `TestData` containing input/label validation windows generated from `validation_start`. |
| `test_data` | GluonTS `TestData` containing input/label test windows generated from `test_start`. |

`test_data` and `val_data` are re-iterable collections of generated window
instances, not arrays of pre-materialized `X` and `Y` tensors. Iteration yields
`(input, label)` pairs. Each input contains the history available at its query
date and each label contains the following forecast horizon.

The number of complete windows is:

```text
floor(interval_length / prediction_length)
```

Consequently, labels never cross the next interval boundary. A remainder
shorter than one forecast horizon is not evaluated. `val_length=0` produces
zero validation windows and accessing `val_data` raises a clear error. A
non-empty validation or test interval shorter than one complete horizon is
rejected instead of silently creating an overlapping window.

These dataset views establish non-overlapping interval boundaries; a
supervised training-window sampler must additionally ensure that every sampled
training label ends at or before `validation_start`.
