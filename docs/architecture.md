# Code architecture

Adaptime keeps TIME's inherited foundation benchmark recognizable while
isolating the proposal pipeline beneath `src/timebench/`.

The inherited baseline path is:

```text
scripts/run_*.sh
  -> experiments/*.py
  -> timebench.evaluation.Dataset and model inference
  -> timebench.pipeline.allocate_run
  -> predictions, metrics, completed schema-1 manifest
  -> scripts/compute_foundation_summary.py
```

The Adaptime path is:

```text
TIME saved-Arrow dataset + dataset YAML
  -> pipeline/runs.py: allocate or reuse one dataset/frequency/term run_n
  -> evaluation/adaptation_data.py: fixed datastore and pre-test/test indices
  -> pipeline/adaptime_extraction.py: V, C, neighbors, Y, N, timed components
  -> pipeline/adaptime_training.py: train statistics and validation selection
  -> pipeline/adaptime_testing.py: frozen V/C/Adaptime TIME comparison
  -> pipeline/adaptime_workflow.py: manifest-selected TIME aggregate
```

Every source row receives a finite-context fraction and an explicit eligibility
reason during extraction. Datastore candidates with incomplete futures are
removed before blockwise search. Retrieval and covariate model calls operate
only on eligible rows; full-size memory maps preserve original TIME row order
and store vanilla output in every fallback slot. Training streams only complete
ridge rows, validation scores ineligible rows as vanilla, and testing decides
fallback without consulting the future target.

`timebench.adaptime` owns exact retrieval and the proposal's readable ridge
math. `timebench.model_loading` owns foundation construction, capability
declarations, offline checkpoints, and the tensor/covariate adapters.
`timebench.evaluation.adaptation_data` owns chronological windows and lazy
Arrow access. `timebench.pipeline` owns disk-backed extraction, model/result
manifests, task-boundary recovery, run selection, and TIME-wide orchestration.
`timebench.scripts` exposes the Python commands. `src/slurm` owns shared
DGX/Selena workflow implementations, while `slurm` contains concise
submit-ready fronts.

One proposal task owns
`outputs/adaptime/tasks/<model>/<target_mode>/<dataset>/<frequency>/<term>/run_n/`.
Preparation, extraction, training, and testing are children of that run. An
interrupted task clears those children and restarts from preparation in the
same `run_n`; completed tasks are selected and reused as units. Aggregate
reports live below `outputs/adaptime/summary/` and record every selected run
manifest. This boundary prevents a partially written extraction or ridge fit
from being treated as a scientific checkpoint.

The inherited `evaluation/window_audit.py` diagnostics inspect the exact TIME
test queries once per distinct model-effective `(L,H)` configuration. Exact
source positions, window events, and full-series features live below shared
`TIME_METADATA`; `feature/features_runner.py` reuses complete artifacts and
repairs only missing source-variate rows. The workflow exports compact
aggregates to its job log tree for standard result synchronization and
publication.

The extraction boundary is deliberate. Large values remain in Arrow or `.npy`
memory maps; distance matrices are bounded by query and datastore blocks;
neighbor foundation forecasts are computed once per selected datastore row;
and every K/alpha candidate reuses the same extraction. Ridge fitting streams
float64 sufficient statistics, so it never materializes the flattened design.
Extraction records test-query representation, retrieval, context construction,
vanilla and covariate model calls, and fixed-datastore preprocessing separately.
Testing adds the ridge design/adjustment cost and derives method-level total and
per-window inference times without treating the fixed training procedure as
online refitting.

The matched external-control path is separate:

```text
completed Adaptime L=512 run
  -> pipeline/tsrag_data.py: matched raw-date budget and TIME test references
  -> external_models/tsrag + model_loading/tsrag.py: pinned ARM/checkpoint
  -> pipeline/tsrag.py: Chronos-T5/FAISS extraction and frozen evaluation
  -> pipeline/tsrag_workflow.py: vanilla/TS-RAG/full-ridge table
```

This preserves TS-RAG's same-series stride-one retrieval and native 64-step
model contract rather than importing proposal code into the external method.

The current proposal path is univariate. Native multivariate evaluation remains
an inherited Chronos-2 control and is not mixed into `full_ridge_shared`.
