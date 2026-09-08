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
  -> Adaptime family                       -> separate TS-RAG control
       fit-grid extraction                      pipeline/tsrag.py extraction
       Ridge selection + Bayesian evidence      external_models/tsrag + loader
       selected-K test extraction               pipeline/tsrag.py inference
       four aligned prediction arrays
  -> evaluation/adaptation.py
       one standard TIME evaluation per method
  -> results/adaptation.py
       comparison over independently completed evaluation manifests
```

`pipeline/adaptime_workflow.py` owns configuration resolution and composes
these phases. The main family and TS-RAG read the same prepared datastore and
official test references, while TS-RAG retains separate extraction and
prediction roots. TS-RAG never loads an Adaptime extraction or fitted model.
The shared datastore retains enough future support for `max(H,64)` and at
least 11 dates per variate so TS-RAG top-10 retrieval remains possible. Ridge
separately applies its `max_k` eligibility gate and becomes vanilla-only when
too few dates survive that gate.

Fit extraction materializes bounded `.npy` arrays for fixed-context
representations, ordered neighbors and distances, forecasts, targets, scales,
eligibility, and timing. Training and validation rows without the complete
backbone context are excluded; official test rows are never excluded.
Neighbor search runs once at `max_k`; every candidate consumes an ordered
prefix on common max-K support, while each `C(K)` remains a distinct backbone
forecast. Closed-form fitting streams float64 sufficient statistics.
When valid primary-`K` training windows do not exceed test windows, the model
artifact records a whole-task vanilla fallback. Sparse validation uses the
primary `K=10`, `alpha=1e-2` without selection when valid validation windows do
not exceed 10% of test windows.

The fitted artifact stores Ridge coefficients and Beta-Bernoulli evidence for
the selected `K`. A trial is one eligible training or validation window;
`C` wins when its per-window MSSE is below `V`, and a tie contributes one half.
The Beta(1,1) posterior mean is the fixed test mixture probability. Test
extraction runs only selected `K`, reuses any neighbor forecasts already
cached during fitting, and computes only newly selected neighbor forecasts.
The prediction artifact contains `V`, hard `C`, `(1-p)V+pC`, and full Ridge;
each adaptation branch falls back to cached `V` on an ineligible test row.

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
prepare
  -> vanilla test pass
  -> fit-grid extraction
  -> Ridge and Bayesian fit
  -> selected-K test extraction
  -> four prediction arrays
  -> four evaluations
  -> report
```

The separate TS-RAG submission schedules its native pipeline independently.
When requested, its report accepts an external full-Ridge root only after
every requested evaluation exactly matches the current identity and scientific
configuration; this never changes TS-RAG execution.

Large series remain in Arrow and large numeric products remain memory-mapped.
`pipeline/runs.py` owns allocation and exact reuse. `src/timebench/scripts/`
contains explicit phase entry points; `src/slurm/run_adaptime_comparison.sh`
is the common DGX/Selena implementation; root `scripts/` compose scheduler
dependencies; `slurm/` contains the concise submit-ready fronts.
