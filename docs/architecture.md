# Code architecture

Adaptime separates data preparation, method-specific extraction, adaptation,
prediction, evaluation, and reporting. Each boundary is independently callable
and owns a configuration-addressed schema-1 run.

```text
TIME saved-Arrow dataset + dataset configuration
  -> evaluation/adaptation_data.py
       shared datastore, train/validation references, official test references
  -> ridge branch                         -> TS-RAG branch
       pipeline/adaptime_extraction.py         pipeline/tsrag.py extraction
       pipeline/adaptime_training.py           external_models/tsrag + loader
       pipeline/adaptation_prediction.py       pipeline/tsrag.py inference
  -> evaluation/adaptation.py
       standard TIME save_window_predictions evaluator
  -> results/adaptation.py
       comparison over independently completed evaluation manifests
```

`pipeline/adaptime_workflow.py` owns configuration resolution and composes
these phases. It does not make TS-RAG depend on a Ridge artifact. Both branches
read the exact same prepared datastore and official test references, while
their extraction and prediction artifacts live under separate method roots.
The shared datastore retains enough future support for `max(H,64)` and at
least 11 dates per variate so TS-RAG top-10 retrieval remains possible. Ridge
separately applies its `max_k` eligibility gate and becomes vanilla-only when
too few dates survive that gate.

Ridge extraction materializes bounded `.npy` arrays for representations,
neighbors, forecasts, targets, eligibility, and timing. Neighbor search may
retain partial results for diagnostics, but a query is RAG-eligible only when
it has `max_k` valid neighbors. Every candidate `K` uses the same eligible rows
and consumes the first `K` neighbors from that shared ordered list. Closed-form
fitting streams float64 sufficient statistics.
When valid primary-`K` training dates do not exceed test dates, the model
artifact records an explicit vanilla fallback instead of solving an empty
ridge. Sparse validation uses the primary `K=10`, `alpha=1e-2` without model
selection when valid validation dates do not exceed 10% of test dates.

TS-RAG owns its Chronos-T5 embedding/FAISS extraction and released ARM
inference. Its reader projects native 512-step contexts and 64-step neighbor
futures from the shared references without creating another datastore.

Both wrappers write deterministic point-prediction artifacts. The common
evaluator exposes each point prediction as the median quantile and delegates
TIME row reconstruction, labels, metrics, and evaluation files to
`evaluation.saver.save_window_predictions`, the same owner used by vanilla
foundation models.

The combined submission order is:

```text
prepare
  +-> ridge pipeline ----+
  +-> TS-RAG pipeline ---+-> report
```

A supplied Ridge-results root is consulted only by the Ridge pipeline and the
final report. Ridge work is skipped only if every requested evaluation exactly
matches the current identity, data fingerprint, pipeline configuration,
experiment configuration, and selected repeat. Otherwise Ridge recomputes in
the local artifact root. TS-RAG execution is unchanged in both cases.

Large series remain in Arrow and large numeric products remain memory-mapped.
`pipeline/runs.py` owns allocation and exact reuse. `src/timebench/scripts/`
contains explicit phase entry points; `src/slurm/run_adaptime_comparison.sh`
is the common DGX/Selena implementation; root `scripts/` compose scheduler
dependencies; `slurm/` contains the concise submit-ready fronts.
