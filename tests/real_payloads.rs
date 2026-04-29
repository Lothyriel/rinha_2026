use std::{
    path::Path,
    sync::{Arc, atomic::AtomicBool},
};

use monoio::{
    io::{AsyncReadRent, AsyncWriteRentExt},
    net::UnixStream,
};
use rinha_2026::{
    detection::FraudEngine,
    model::{FraudScoreRequest, FraudScoreResponse},
    server::{self, AppState},
};

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
    sonic_rs::from_str(include_str!("fixtures/example-payloads.json"))
        .expect("fixture payloads should deserialize")
}

#[monoio::test(driver = "fusion")]
async fn official_payload_samples_return_valid_scores() {
    let state = AppState {
        engine: load_engine(false),
    };
    let shutdown = Arc::new(AtomicBool::new(false));
    let (mut client, server_stream) = UnixStream::pair().expect("pair should open");

    monoio::spawn(server::serve_connection(server_stream, state, shutdown));

    for payload in load_payloads() {
        let payload_id = payload.id.clone();
        let request = build_json_request(&payload, false);
        let response = round_trip(&mut client, request).await;
        assert_eq!(response.status_code, 200, "payload id {payload_id}");

        let body: FraudScoreResponse = sonic_rs::from_slice(&response.body).expect("response json");

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

#[monoio::test(driver = "fusion")]
async fn invalid_timestamp_returns_bad_request() {
    let state = AppState {
        engine: load_engine(true),
    };
    let shutdown = Arc::new(AtomicBool::new(false));
    let (mut client, server_stream) = UnixStream::pair().expect("pair should open");
    monoio::spawn(server::serve_connection(server_stream, state, shutdown));

    let mut payload = load_payloads().remove(0);
    payload.transaction.requested_at = "not-a-timestamp".to_owned();

    let response = round_trip(&mut client, build_json_request(&payload, true)).await;
    assert_eq!(response.status_code, 400);
    assert!(response.connection_close);
}

#[monoio::test(driver = "fusion")]
async fn keep_alive_handles_multiple_requests_on_same_connection() {
    let state = AppState {
        engine: load_engine(true),
    };
    let shutdown = Arc::new(AtomicBool::new(false));
    let (mut client, server_stream) = UnixStream::pair().expect("pair should open");
    monoio::spawn(server::serve_connection(server_stream, state, shutdown));

    let payload = load_payloads().remove(0);
    let first = round_trip(&mut client, build_json_request(&payload, false)).await;
    let second = round_trip(&mut client, build_json_request(&payload, false)).await;

    assert_eq!(first.status_code, 200);
    assert_eq!(second.status_code, 200);
    assert!(!first.connection_close);
    assert!(!second.connection_close);
}

fn build_json_request(payload: &FraudScoreRequest, close_connection: bool) -> Vec<u8> {
    let body = sonic_rs::to_vec(payload).expect("json body");
    let connection = if close_connection {
        "close"
    } else {
        "keep-alive"
    };

    format!(
        "POST /fraud-score HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: {connection}\r\n\r\n",
        body.len()
    )
    .into_bytes()
    .into_iter()
    .chain(body)
    .collect()
}

struct ParsedResponse {
    status_code: u16,
    body: Vec<u8>,
    connection_close: bool,
}

async fn round_trip(stream: &mut UnixStream, request: Vec<u8>) -> ParsedResponse {
    let (result, _) = stream.write_all(request).await;
    result.expect("request should write");
    read_response(stream).await
}

async fn read_response(stream: &mut UnixStream) -> ParsedResponse {
    let mut pending = Vec::with_capacity(4096);
    let mut read_buf = Vec::with_capacity(4096);

    loop {
        if let Some(response) = try_parse_response(&pending) {
            return response;
        }

        let (result, buffer) = stream.read(read_buf).await;
        read_buf = buffer;
        let read = result.expect("response should read");
        assert!(read > 0, "server closed connection early");
        pending.extend_from_slice(&read_buf[..read]);
        read_buf.clear();
    }
}

fn try_parse_response(buffer: &[u8]) -> Option<ParsedResponse> {
    let headers_end = buffer.windows(4).position(|window| window == b"\r\n\r\n")?;
    let header_text = std::str::from_utf8(&buffer[..headers_end]).ok()?;
    let mut lines = header_text.split("\r\n");
    let status_line = lines.next()?;
    let status_code = status_line.split(' ').nth(1)?.parse().ok()?;
    let mut content_length = 0usize;
    let mut connection_close = false;

    for line in lines {
        let (name, value) = line.split_once(':')?;
        if name.eq_ignore_ascii_case("Content-Length") {
            content_length = value.trim().parse().ok()?;
        } else if name.eq_ignore_ascii_case("Connection") {
            connection_close = value.trim().eq_ignore_ascii_case("close");
        }
    }

    let body_start = headers_end + 4;
    let body_end = body_start + content_length;
    if buffer.len() < body_end {
        return None;
    }

    Some(ParsedResponse {
        status_code,
        body: buffer[body_start..body_end].to_vec(),
        connection_close,
    })
}
