# Code architecture

Adaptime separates data preparation, method-specific extraction, adaptation,
prediction, evaluation, and reporting. Each boundary is independently callable
and owns a configuration-addressed schema-1 run.

```text
TIME saved-Arrow dataset + dataset configuration
  -> evaluation/adaptation_data.py
       shared datastore, train/validation references, official test references
  -> pipeline/adaptime_vanilla.py
       flexible-context vanilla forecast for every official test row
  -> pipeline/adaptime_cache.py
       reusable source-window backbone forecasts shared by adaptors
  -> fixed Adaptime family
       fit-grid extraction -> Ridge/Bayesian selection -> selected-K prediction
  -> rolling horizon Ridge
       causal datastore -> incremental per-variate/per-horizon fit
  -> independent TS-RAG control
       pipeline/tsrag.py extraction -> external ARM inference
  -> evaluation/adaptation.py
       one standard TIME evaluation per method
  -> results/adaptation.py
       unified comparison over independently completed evaluation manifests
```

`pipeline/adaptime_workflow.py` owns configuration resolution and composes
these phases. The main family and TS-RAG read the same prepared datastore and
official test references, while TS-RAG retains separate extraction and
prediction roots. TS-RAG never loads an Adaptime extraction or fitted model.
The shared datastore retains enough future support for `max(H,64)`. TS-RAG
admits it only when every variate has at least 11 dates, so top-10 retrieval
remains possible. Ridge separately applies its `max_k` eligibility gate and
becomes vanilla-only when too few dates survive that gate. Shared preparation
records the global minimum; the project-owned TS-RAG data adapter validates it
for both new and reused manifests before TS-RAG representation extraction.
Generic preparation and the Ridge fallback path remain method-neutral.

Fit extraction materializes bounded `.npy` arrays for fixed-retrieval-context
representations, ordered neighbors and distances, forecasts, targets, scales,
eligibility, and timing. Training and validation rows without the complete
configured retrieval context are excluded; foundation predictions independently
use the history available at each origin up to the backbone limit. Official
test rows are never excluded. Task-specific alignment, fitting/datastore
strides, rolling strides, and retrieval lookbacks are resolved from
`adaptime_tasks` in the dataset configuration.
Neighbor search runs once at `max_k`; every candidate consumes an ordered
prefix on common max-K support, while each `C(K)` remains a distinct backbone
forecast. Closed-form fitting streams float64 sufficient statistics.
When valid primary-`K` training windows do not exceed test windows, the model
artifact records a whole-task vanilla fallback. Sparse validation uses the
primary `K=10`, `alpha=1e-2` without selection when valid validation windows do
not exceed 10% of test windows.

The fitted artifact always includes virtual vanilla (`K=0`) and stores global
or per-variate Ridge coefficients plus Beta-Bernoulli evidence for each
eligible positive `K`. A trial is one eligible adaptation-training
window; `C` wins when its per-window MSSE is below `V`, and a tie contributes
one half. Adaptation validation compares the resulting frozen mixtures and
selects the Bayesian `K`, or virtual vanilla, by MSSE. The selected Beta(1,1)
posterior mean is the fixed test mixture probability. Candidate losses are
averaged by validation date across variates, then a paired moving-date-block
bootstrap applies a one-standard-error rule. Statistically indistinguishable
candidates prefer vanilla, then lower `K`, then stronger regularization. Test
extraction runs only selected `K`, reuses any neighbor forecasts already cached
during fitting, and computes only newly selected neighbor forecasts.
The prediction artifact contains `V`, hard `C`, both Bayesian candidates,
`V+C`, `V+Y_1..Y_K`, shared and per-variate full Ridge, and the overall
validation-selected method. Each branch falls back to cached `V` when vanilla
wins validation or its test row is ineligible.

`pipeline/adaptime_rolling.py` owns the independent
`rolling_y_ridge_horizon` method. At each official query it uses 64--100
same-variate fitting dates aligned to the query's retrieval-period phase and a
fixed-size causal datastore balanced across variates. Incremental sufficient
statistics fit one `V+Y_1..Y_K` coefficient vector per variate and horizon;
unsupported queries remain vanilla. This method has fixed `K` and alpha and no
validation pass.

TS-RAG owns its Chronos-T5 embedding/FAISS extraction and released ARM
inference. Its reader projects native 512-step contexts and 64-step neighbor
futures from the shared references without creating another datastore.

All methods write deterministic point-prediction artifacts. The common
evaluator exposes each point prediction as the median quantile and delegates
TIME row reconstruction, labels, metrics, and evaluation files to
`evaluation.saver.save_window_predictions`, the same owner used by vanilla
foundation models.

The combined submission order is:

```text
shared Evaluating-TSFMs Seasonal Naive grid (prerequisite)
  -> prepare -> vanilla test pass
       -> fit-grid extraction -> Ridge and Bayesian fit
          -> selected-K test extraction -> family predictions/evaluations
       -> independent TS-RAG extraction/prediction/evaluation
  -> unified report after both branches succeed
```

The TS-RAG-only submission remains available and schedules its native pipeline
independently. When requested, its report accepts an external full-Ridge root
only after every requested evaluation exactly matches the current identity and
scientific configuration; this never changes TS-RAG execution.

Large series remain in Arrow and large numeric products remain memory-mapped.
The shared forecast cache uses a readable model/mode/task hierarchy and
ordinary `run_n` allocation. Completed runs are reused only when their plain
configuration matches. It stores source-window backbone forecasts in coarse
uncompressed shards, writes the manifest only at lifecycle boundaries, and
records discovery, lookup, read, compute, write, and manifest timings.
Retrieval representations are recomputed, while canonical test vanilla and
composite covariate forecasts remain in their phase artifacts. Rolling and
fixed neighbor tables remain separate.
`pipeline/runs.py` owns allocation and exact reuse. `src/timebench/scripts/`
contains explicit phase entry points; `src/slurm/run_adaptime_comparison.sh`
is the common DGX/Selena implementation; root `scripts/` compose scheduler
dependencies; `slurm/` contains the concise submit-ready fronts.

Seasonal Naive defines the common metric grid. Every foundation forecast and
canonical vanilla must be finite on its expected target steps. A non-finite
fixed candidate, rolling Ridge, or TS-RAG forecast on that grid is replaced as
one complete cell by canonical vanilla, and its per-window mask and counts are
retained. `results/adaptation.py` requires matching grid metadata, preserves
lifecycle labels, applies repeat-then-configuration averaging when requested,
and reports finite/grid/total metric counts plus non-finite fallback counts.

`src/timebench/scripts/time_inference.py` is a separate measurement path. Its parent process
selects shared random test references and launches a fresh child process for
each method/example pair. Children may load frozen fitted or datastore state,
but recompute all query-side work and never consume test prediction, neighbor,
or representation caches. `src/slurm/time_inference.sh` owns the common cluster
command used by the identically named DGX and Selena fronts.
