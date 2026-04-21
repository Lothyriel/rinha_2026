use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};

use crate::{
    detection::{FraudEngine, FraudEngineError},
    model::{FraudScoreRequest, FraudScoreResponse},
};

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<FraudEngine>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/ready", get(ready))
        .route("/fraud-score", post(fraud_score))
        .with_state(state)
}

async fn ready() -> StatusCode {
    StatusCode::OK
}

async fn fraud_score(
    State(state): State<AppState>,
    Json(payload): Json<FraudScoreRequest>,
) -> Result<Json<FraudScoreResponse>, AppError> {
    let response = state.engine.score(&payload)?;
    Ok(Json(response))
}

pub enum AppError {
    BadRequest(String),
    Unavailable(String),
}

impl From<FraudEngineError> for AppError {
    fn from(value: FraudEngineError) -> Self {
        match value {
            FraudEngineError::InvalidRequest(message) => Self::BadRequest(message),
            FraudEngineError::Unavailable(message) | FraudEngineError::Load(message) => {
                Self::Unavailable(message)
            }
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AppError::BadRequest(message) => (StatusCode::BAD_REQUEST, message),
            AppError::Unavailable(message) => (StatusCode::SERVICE_UNAVAILABLE, message),
        };

        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}
