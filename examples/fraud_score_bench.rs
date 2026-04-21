use std::{hint::black_box, path::Path};

use rinha_2026::{
    detection::{FraudEngine, SearchBackendKind},
    model::FraudScoreRequest,
};

fn load_engine() -> FraudEngine {
    FraudEngine::load(Path::new("spec/resources"), SearchBackendKind::Exact)
        .expect("engine should load spec resources")
}

fn load_payloads() -> Vec<FraudScoreRequest> {
    serde_json::from_str(include_str!("../tests/fixtures/example-payloads.json"))
        .expect("fixture payloads should deserialize")
}

fn main() {
    let engine = load_engine();
    let payloads = load_payloads();

    for index in 0..500usize {
        let payload = &payloads[index % payloads.len()];
        let response = engine
            .score(black_box(payload))
            .expect("warmup should succeed");
        black_box(response);
    }

    let iterations = 5_000usize;
    let started_at = std::time::Instant::now();

    for index in 0..iterations {
        let payload = &payloads[index % payloads.len()];
        let response = engine
            .score(black_box(payload))
            .expect("scoring should succeed");
        black_box(response);
    }

    let elapsed = started_at.elapsed();
    let total_ms = elapsed.as_secs_f64() * 1_000.0;
    let us_per_op = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();

    println!(
        "fraud_score_exact iterations={iterations} total_ms={total_ms:.3} us_per_op={us_per_op:.3} ops_per_sec={ops_per_sec:.2}"
    );
}
