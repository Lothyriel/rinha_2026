use std::sync::Arc;

use crate::{
    detection::{FraudEngine, FraudEngineError},
    model::{FraudScoreRequest, FraudScoreResponse},
};

use super::http1::{self, ResponseBytes};

#[derive(serde::Serialize)]
struct ErrorResponse<'a> {
    error: &'a str,
}

pub fn handle_fraud_score(
    body: &[u8],
    engine: &Arc<FraudEngine>,
    connection_close: bool,
) -> ResponseBytes {
    let payload = match sonic_rs::from_slice::<FraudScoreRequest>(body) {
        Ok(payload) => payload,
        Err(error) => {
            return error_response(
                400,
                &format!("invalid json payload: {error}"),
                connection_close,
            );
        }
    };

    match engine.score(&payload) {
        Ok(response) => success_response(&response, connection_close),
        Err(error) => match error {
            FraudEngineError::InvalidRequest(message) => {
                error_response(400, &message, connection_close)
            }
            FraudEngineError::Unavailable(message) | FraudEngineError::Load(message) => {
                error_response(503, &message, connection_close)
            }
        },
    }
}

fn success_response(response: &FraudScoreResponse, connection_close: bool) -> ResponseBytes {
    match sonic_rs::to_vec(response) {
        Ok(body) => http1::json_response(200, body, connection_close),
        Err(error) => error_response(500, &format!("failed to serialize response: {error}"), true),
    }
}

fn error_response(status: u16, message: &str, connection_close: bool) -> ResponseBytes {
    let payload = ErrorResponse { error: message };
    match sonic_rs::to_vec(&payload) {
        Ok(body) => http1::json_response(status, body, connection_close),
        Err(_) => http1::json_response(
            status,
            format!(r#"{{"error":"{message}"}}"#).into_bytes(),
            true,
        ),
    }
}
