# Example decision — lazy model lifecycle

tags: #decision #architecture #resources

**Date:** 2026-01-01
**Status:** accepted

## Context

The machine has 16 GB of RAM and a GPU shared with other workloads (training,
speech, video). Keeping a 0.6B embedding model resident costs roughly 1.2 GB of
RAM and, on the GPU path, ~2.5 GB of VRAM.

## Decision

The embedding model is loaded **per command** and the process exits afterwards.
Vectors are cached in SQLite, so the model is only needed when documents change or
when a search falls back to the semantic stream.

## Consequences

- Idle memory cost: zero.
- Reindex latency: model load time (~2–5 s on CPU) is paid only when something
  changed.
- Keyword search usually answers without loading the model at all.

Related: [[setups/example-setup]] · [[00-index]]
