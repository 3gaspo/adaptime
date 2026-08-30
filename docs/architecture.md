# Code architecture

This document owns Adaptime's public source-responsibility map and executable
data flow.

Adaptime currently retains the inherited TIME benchmark layout. The
retrieval, adaptor-training, model-loading, evaluation, and reporting owners
have not yet been fixed, so no proposed package boundaries or runtime stages
are documented here yet.

When the implementation stabilizes, this page will identify each source owner,
show the path from datasets and query windows through retrieval and prediction
to metrics, and state the boundaries between inherited TIME code, adapted
external methods, and the Adaptime proposal.
