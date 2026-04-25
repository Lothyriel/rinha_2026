use std::{path::Path, sync::Arc};

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use rinha_2026::{
    app::{AppState, router},
    detection::FraudEngine,
    model::{FraudScoreRequest, FraudScoreResponse},
};
use tower::ServiceExt;

fn load_engine(example: bool) -> Arc<FraudEngine> {
    let path = Path::new("spec/resources");

    let engine = if example {
        FraudEngine::load_example(path)
    } else {
        FraudEngine::load(path)
    };

    Arc::new(engine.expect("engine should load spec resources"))
}

fn load_payloads() -> Vec<FraudScoreRequest> {
    serde_json::from_str(include_str!("fixtures/example-payloads.json"))
        .expect("fixture payloads should deserialize")
}

#[tokio::test]
async fn official_payload_samples_return_valid_scores() {
    let app = router(AppState {
        engine: load_engine(false),
    });

    for payload in load_payloads() {
        let payload_id = payload.id.clone();
        let request = Request::builder()
            .method("POST")
            .uri("/fraud-score")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&payload).expect("json body")))
            .expect("request should build");

        let response = app
            .clone()
            .oneshot(request)
            .await
            .expect("request should succeed");
        assert_eq!(response.status(), StatusCode::OK, "payload id {payload_id}");

        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should read");
        let body: FraudScoreResponse = serde_json::from_slice(&bytes).expect("response json");

        assert!(
            (0.0..=1.0).contains(&body.fraud_score),
            "payload id {payload_id}"
        );
        assert_eq!(
            body.approved,
            body.fraud_score < 0.6,
            "payload id {payload_id}"
        );
    }
}

#[tokio::test]
async fn invalid_timestamp_returns_bad_request() {
    let app = router(AppState {
        engine: load_engine(true),
    });
    let mut payload = load_payloads().remove(0);
    payload.transaction.requested_at = "not-a-timestamp".to_owned();

    let request = Request::builder()
        .method("POST")
        .uri("/fraud-score")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).expect("json body")))
        .expect("request should build");

    let response = app.oneshot(request).await.expect("request should succeed");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}
