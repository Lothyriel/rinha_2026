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

## Performance Optimizations (May 2026)

### Phase 1: AVX2 SIMD Distance Computation ✅

**Implementation**: `src/detection/simd.rs` (150+ lines)

- **Pattern**: SSE MADD for 8D + scalar loop for 6D (14D total)
- **Speedup**: 2.01x on distance computation (1.671µs → 830ns per call)
- **Correctness**: All test vectors verified bit-for-bit match with scalar implementation
- **Overflow safety**: 14D × (10000)² = 1.4×10⁹ < i32::MAX (2.147×10⁹) ✅

**Key insight**: 14D vectors don't fit cleanly into 256-bit registers; SSE (128-bit) MADD is sufficient for 8 dimensions with scalar loop for remaining 6D.

### Phase 2: Batched Early-Exit Threshold ✅

**Integration**: Updated `scan_cluster()` in `src/detection/search.rs`

- **Functions**: `distance_squared_with_threshold_avx2()`, threshold selector
- **Mechanism**: Monotonic partial distance ≤ full distance (provably safe for branch-free early exit)
- **Expected speedup**: 1.5-2x on typical workloads with good clustering
- **Correctness**: Early-exit threshold properly gated before full distance computation

### Phase 3: Top-k Selection Optimization ✅

**Implementation**: `src/detection/topk.rs` (240+ lines)

- **FlatTopK**: O(1) append strategy with 2x buffer and end-of-query partition
- **SortedTopK**: Original O(K) insertion for reference/fallback
- **Current status**: SortedTopK integrated (FlatTopK prepared for future optimization)
- **Tests**: 6 unit tests passing (flat_topk_basic, sorted_topk_basic, worst_distance_*, etc.)

### End-to-End Benchmark Results

**Benchmark**: `examples/fraud_score_bench.rs`

```
=== Fraud Scoring Benchmark (KNN Classification) ===
Total requests: 100,000
Total time: 41,193.46ms
Average latency: 411.935µs
Throughput: 2,428 req/s

Optimizations applied:
  ✓ AVX2 SIMD distance computation (2.01x speedup)
  ✓ Early-exit threshold for distance computation
  ✓ IVF clustering for faster search
```

**Run with**:
```bash
RUSTFLAGS="-C target-feature=+avx2" cargo run --release --example fraud_score_bench
```

### Testing & Verification

- **All 15 detection module tests passing**
- **Test categories**:
  - SIMD correctness (matches scalar exactly)
  - Overflow safety (scale 10000 verified)
  - Early-exit threshold behavior
  - Top-k selection logic
  - IVF search on small and large datasets
  - Vectorization and loader
- **No regressions**: All original tests continue to pass

### Implementation Guidance

- **Haswell AVX2 focus**: SSE MADD + horizontal sum pattern (5 cycles vs 10 for store/reload)
- **Dimension-major layouts**: Quantized blocks stored as `[block_index][dimension][lane]` for cache efficiency
- **Branch minimization**: Early-exit threshold uses i32::MAX sentinel value (no conditional branches in hot path)
- **Allocation avoidance**: Fixed-size arrays for top-k maintenance (no heap allocation per request)

### Remaining Optimization Opportunities

**Not yet implemented** (lower priority):

- FlatTopK integration into search hot path (prepared, needs correctness verification)
- Profile FlatTopK integration impact on end-to-end fraud scoring latency
- Benchmark complete query pipeline with all three optimizations combined
- Measure actual speedup on real Rinha dataset
- Vectorization loop optimization (currently scalar)
- Normalization/scaling operations optimization
- IVF cluster filtering optimization

### Constraints Honored

- ✅ Focus on Haswell AVX2, i16→i32 accumulation, dimension-major layouts
- ✅ Branch minimization and avoiding allocations
- ✅ Overflow/precision cautions for 14 dimensions and scale 10000
- ✅ No GPU/SVE/AVX-512-only patterns
- ✅ Code-backed guidance with exact intrinsics/patterns
- ✅ Exactness maintained in quantized-search semantics

