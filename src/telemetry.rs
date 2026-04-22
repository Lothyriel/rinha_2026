use std::time::Duration;

use metrics::{Unit, describe_counter, describe_histogram};
use metrics_util::Quantile;

const DEFAULT_EXPORT_INTERVAL_SECS: u64 = 10;

fn describe_metrics() {
    describe_counter!("http_reqs", Unit::Count, "Total HTTP /fraud-score requests");
    describe_histogram!(
        "http_handler_time",
        Unit::Microseconds,
        "HTTP /fraud-score handler latency"
    );
    describe_histogram!(
        "score_engine",
        Unit::Microseconds,
        "Fraud engine latency grouped by step"
    );
}

pub fn install_from_env() -> Result<(), String> {
    let use_metrics = std::env::var("METRICS").is_ok();

    if !use_metrics {
        return Ok(());
    };

    let export_interval_secs = std::env::var("RINHA_TRACE_EXPORT_INTERVAL_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_EXPORT_INTERVAL_SECS);

    let mut recorder = metrics_printer::PrintRecorder::default();

    recorder
        .do_print_metadata()
        .set_print_interval(Duration::from_secs(export_interval_secs))
        .select_quantiles(Box::new([
            Quantile::new(0.50),
            Quantile::new(0.99),
            Quantile::new(1.0),
        ]));

    recorder.install().expect("register recorder");

    describe_metrics();

    tracing::info!(export_interval_secs, "metrics exporter enabled");

    Ok(())
}
