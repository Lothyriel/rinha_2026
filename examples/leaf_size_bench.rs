use std::{hint::black_box, path::Path, time::Instant};

use rinha_2026::{
    detection::FraudEngine,
    model::{FraudScoreRequest, FraudScoreResponse},
};

const DEFAULT_LEAF_SIZES: &[usize] = &[1, 2, 4, 6, 8, 16, 24, 32, 64, 128];

fn main() {
    let payloads = load_payloads();
    let iterations = std::env::var("RINHA_BENCH_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(5_000);

    let leaf_sizes = DEFAULT_LEAF_SIZES.to_vec();

    println!(
        "leaf_size_bench dataset={} payloads={} iterations={}",
        Path::new("spec/resources/references.json.gz").display(),
        payloads.len(),
        iterations,
    );

    let baseline_outputs = collect_outputs(
        &FraudEngine::load_with_leaf_size(Path::new("spec/resources"), 32)
            .expect("baseline engine should load"),
        &payloads,
    );

    for leaf_size in leaf_sizes {
        run_leaf_size_benchmark(leaf_size, &payloads, &baseline_outputs, iterations);
    }
}

fn run_leaf_size_benchmark(
    leaf_size: usize,
    payloads: &[FraudScoreRequest],
    baseline_outputs: &[FraudScoreResponse],
    iterations: usize,
) {
    let build_started_at = Instant::now();
    let engine = FraudEngine::load_with_leaf_size(Path::new("spec/resources"), leaf_size)
        .expect("engine should load spec resources");
    let build_elapsed = build_started_at.elapsed();

    let candidate_outputs = collect_outputs(&engine, payloads);
    let correctness = compare_outputs(baseline_outputs, &candidate_outputs);

    for index in 0..500usize {
        let payload = &payloads[index % payloads.len()];
        let response = engine
            .score(black_box(payload))
            .expect("warmup should succeed");
        black_box(response);
    }

    let started_at = Instant::now();
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
        "leaf_size_bench leaf_size={leaf_size} build_ms={build_ms:.3} iterations={iterations} total_ms={total_ms:.3} us_per_op={us_per_op:.3} ops_per_sec={ops_per_sec:.2} same_approved_pct={same_approved_pct:.2} mean_abs_score_error={mean_abs_score_error:.6} max_abs_score_error={max_abs_score_error:.6}",
        same_approved_pct = correctness.same_approved_pct,
        mean_abs_score_error = correctness.mean_abs_score_error,
        max_abs_score_error = correctness.max_abs_score_error,
    );
}

fn load_payloads() -> Vec<FraudScoreRequest> {
    serde_json::from_str(include_str!("../tests/fixtures/example-payloads.json"))
        .expect("fixture payloads should deserialize")
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
    baseline_outputs: &[FraudScoreResponse],
    candidate_outputs: &[FraudScoreResponse],
) -> CorrectnessSummary {
    assert_eq!(baseline_outputs.len(), candidate_outputs.len());

    let mut same_approved = 0usize;
    let mut total_abs_score_error = 0.0f64;
    let mut max_abs_score_error = 0.0f64;

    for (baseline, candidate) in baseline_outputs.iter().zip(candidate_outputs.iter()) {
        if baseline.approved == candidate.approved {
            same_approved += 1;
        }

        let abs_score_error = (baseline.fraud_score - candidate.fraud_score).abs() as f64;
        total_abs_score_error += abs_score_error;
        max_abs_score_error = max_abs_score_error.max(abs_score_error);
    }

    let total = baseline_outputs.len() as f64;

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
