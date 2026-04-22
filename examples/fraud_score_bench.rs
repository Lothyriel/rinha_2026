use std::{hint::black_box, path::Path};

use rinha_2026::{
    detection::{FraudEngine, HnswConfig, SearchBackendKind},
    model::{FraudScoreRequest, FraudScoreResponse},
};

fn load_engine(search_backend: SearchBackendKind, hnsw_config: HnswConfig) -> FraudEngine {
    FraudEngine::load(Path::new("spec/resources"), search_backend, hnsw_config)
        .expect("engine should load spec resources")
}

fn load_payloads() -> Vec<FraudScoreRequest> {
    serde_json::from_str(include_str!("../tests/fixtures/example-payloads.json"))
        .expect("fixture payloads should deserialize")
}

fn main() {
    let payloads = load_payloads();
    let iterations = 5_000usize;
    let build_efforts = [50, 100, 200];
    let search_efforts = [2, 8, 32];
    let exact_outputs = collect_outputs(
        &load_engine(SearchBackendKind::Exact, HnswConfig::default()),
        &payloads,
    );

    run_benchmark(
        "exact",
        SearchBackendKind::Exact,
        HnswConfig::default(),
        &payloads,
        &exact_outputs,
        iterations,
    );

    for build_effort in build_efforts {
        for search_effort in search_efforts {
            run_benchmark(
                &format!("hnsw ef_construction={build_effort} ef_search={search_effort}"),
                SearchBackendKind::Hnsw,
                HnswConfig {
                    ef_construction: build_effort,
                    ef_search: search_effort,
                },
                &payloads,
                &exact_outputs,
                iterations,
            );
        }
    }
}

fn run_benchmark(
    label: &str,
    search_backend: SearchBackendKind,
    hnsw_config: HnswConfig,
    payloads: &[FraudScoreRequest],
    exact_outputs: &[FraudScoreResponse],
    iterations: usize,
) {
    let build_started_at = std::time::Instant::now();
    let engine = load_engine(search_backend, hnsw_config);
    let build_elapsed = build_started_at.elapsed();
    let benchmark_outputs = collect_outputs(&engine, payloads);
    let correctness = compare_outputs(exact_outputs, &benchmark_outputs);

    for index in 0..500usize {
        let payload = &payloads[index % payloads.len()];
        let response = engine
            .score(black_box(payload))
            .expect("warmup should succeed");
        black_box(response);
    }

    let started_at = std::time::Instant::now();

    for index in 0..iterations {
        let payload = &payloads[index % payloads.len()];
        let response = engine
            .score(black_box(payload))
            .expect("scoring should succeed");
        black_box(response);
    }

    let elapsed = started_at.elapsed();
    let build_ms = build_elapsed.as_secs_f64() * 1_000.0;
    let total_ms = elapsed.as_secs_f64() * 1_000.0;
    let us_per_op = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();

    println!(
        "fraud_score_bench label={label} build_ms={build_ms:.3} iterations={iterations} total_ms={total_ms:.3} us_per_op={us_per_op:.3} ops_per_sec={ops_per_sec:.2} same_approved_pct={same_approved_pct:.2} mean_abs_score_error={mean_abs_score_error:.6} max_abs_score_error={max_abs_score_error:.6}",
        same_approved_pct = correctness.same_approved_pct,
        mean_abs_score_error = correctness.mean_abs_score_error,
        max_abs_score_error = correctness.max_abs_score_error,
    );
}

fn collect_outputs(
    engine: &FraudEngine,
    payloads: &[FraudScoreRequest],
) -> Vec<FraudScoreResponse> {
    payloads
        .iter()
        .map(|payload| engine.score(payload).expect("scoring should succeed"))
        .collect()
}

fn compare_outputs(
    exact_outputs: &[FraudScoreResponse],
    candidate_outputs: &[FraudScoreResponse],
) -> CorrectnessSummary {
    assert_eq!(exact_outputs.len(), candidate_outputs.len());

    let mut same_approved = 0usize;
    let mut total_abs_score_error = 0.0f64;
    let mut max_abs_score_error = 0.0f64;

    for (exact, candidate) in exact_outputs.iter().zip(candidate_outputs.iter()) {
        if exact.approved == candidate.approved {
            same_approved += 1;
        }

        let abs_score_error = (exact.fraud_score - candidate.fraud_score).abs() as f64;
        total_abs_score_error += abs_score_error;
        max_abs_score_error = max_abs_score_error.max(abs_score_error);
    }

    let total = exact_outputs.len() as f64;

    CorrectnessSummary {
        same_approved_pct: (same_approved as f64 / total) * 100.0,
        mean_abs_score_error: total_abs_score_error / total,
        max_abs_score_error,
    }
}

struct CorrectnessSummary {
    same_approved_pct: f64,
    mean_abs_score_error: f64,
    max_abs_score_error: f64,
}
