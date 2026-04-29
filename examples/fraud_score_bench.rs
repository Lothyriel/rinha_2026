use std::{hint::black_box, path::Path, time::Instant};

use rinha_2026::{detection::FraudEngine, model::FraudScoreRequest};

fn main() {
    let payloads = load_payloads();
    let iterations = std::env::var("RINHA_BENCH_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(20_000);

    let engine = FraudEngine::load(Path::new("spec/resources")).expect("engine should load");

    for index in 0..1_000usize {
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

    let us_per_op = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();

    println!(
        "fraud_score_bench references={} payloads={} iterations={} us_per_op={us_per_op:.3} ops_per_sec={ops_per_sec:.2}",
        engine.reference_count(),
        payloads.len(),
        iterations,
    );
}

fn load_payloads() -> Vec<FraudScoreRequest> {
    sonic_rs::from_str(include_str!("../tests/fixtures/example-payloads.json"))
        .expect("fixture payloads should deserialize")
}
