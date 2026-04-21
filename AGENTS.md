# AGENTS.md

## Repository implementation choices

### Goal

This repository implements the Rinha de Backend 2026 fraud scoring API in Rust, prioritizing:

- correctness against the published spec
- low operational complexity
- predictable performance under tight CPU/memory limits

### API shape

- `GET /ready` returns `200 OK` when the app is ready
- `POST /fraud-score` receives the official transaction payload and returns:
  - `approved: bool`
  - `fraud_score: f32`
- runtime listener is configurable: TCP on `PORT` by default, or Unix socket via `RINHA_UNIX_SOCKET_PATH` on Unix hosts

### Application structure

- `src/main.rs`
  - process startup
  - config loading from env
  - graceful shutdown wiring
- `src/app.rs`
  - shared Axum router and handlers
- `src/detection.rs`
  - vectorization
  - dataset loading
  - nearest-neighbor search
  - fraud decision logic
- `src/model.rs`
  - request/response models
- `examples/fraud_score_bench.rs`
  - simple in-process benchmark harness

### Fraud detection choices

- Implementation follows the published 14-dimension vectorization approach
- Default search backend is **exact KNN** with squared Euclidean distance
- `K = 5`
- `fraud_score = fraud_votes / 5`
- `approved = fraud_score < 0.6`

### Search backend strategy

Two backends exist:

#### 1. `exact` (default)

Chosen as default because the challenge spec is based on exact vector search semantics.

- brute-force scan over the loaded references
- fixed-size top-k maintenance in arrays
- no heap allocation per request for nearest-neighbor selection

#### 2. `hnsw` (experimental)

Available through:

```bash
RINHA_SEARCH_BACKEND=hnsw
```

Notes:

- implemented with `vicinity`
- kept optional on purpose
- if HNSW build/search fails, code falls back to `exact`
- not the default because approximate search may hurt challenge score correctness

### Data loading choices

Primary runtime resources:

- `resources/references.json.gz`
- `resources/mcc_risk.json`
- `resources/normalization.json`

Fallback resource:

- `resources/example-references.json`

Design choice:

- load all references eagerly at startup
- keep reference vectors in memory for fast request-time scoring

### Graceful shutdown

The server handles:

- `SIGTERM`
- `CTRL+C`

This was added because container orchestration was previously forcing hard kills during teardown.

### Container/runtime choices

- Rust multi-stage build in `Dockerfile`
- final runtime image based on `debian:bookworm-slim`
- nginx load balancer in front of two API instances
- compose defaults to `linux/amd64`
- healthchecks added for API and load balancer
- API image publication is tag-driven via GitHub Actions to GHCR
- current compose/nginx wiring still uses TCP between containers; Unix socket mode is available for direct process execution or future shared-volume setups

### Testing choices

Current automated validation includes:

- unit tests for vectorization behavior
- integration tests using real official payload samples
- invalid-input test for bad timestamp handling

Real payload fixtures are stored in:

- `tests/fixtures/example-payloads.json`

### Benchmarking choices

Current benchmark is intentionally simple:

- in-process benchmark of `engine.score()`
- uses real payload samples
- intended for quick throughput regression checks

Run with:

```bash
cargo run --release --example fraud_score_bench
```

### Non-goals so far

Not implemented yet:

- full validation against all entries from upstream `test/test-data.json`
- HTTP-level benchmark harness for `/fraud-score`
- advanced memory layout tuning
- stronger ANN benchmarking/tuning for HNSW

### Guidance for future changes

- preserve exact backend as default unless benchmark evidence proves otherwise
- prefer small, measurable performance changes over large refactors
- keep behavior aligned with upstream examples and expected response shape
- treat challenge score correctness as more important than clever architecture
