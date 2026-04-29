use std::{fmt, str};

pub const MAX_BUFFERED_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 32 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    Ready,
    FraudScore,
    NotFound,
    MethodNotAllowed,
}

#[derive(Debug)]
pub struct ParsedRequest {
    pub route: Route,
    pub body_start: usize,
    pub body_end: usize,
    pub total_len: usize,
    pub connection_close: bool,
}

#[derive(Debug)]
pub struct ParseError {
    message: &'static str,
}

impl ParseError {
    fn new(message: &'static str) -> Self {
        Self { message }
    }

    pub fn message(&self) -> &'static str {
        self.message
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message)
    }
}

pub struct ResponseBytes {
    pub bytes: Vec<u8>,
    pub keep_alive: bool,
}

pub fn try_parse_request(buffer: &[u8]) -> Result<Option<ParsedRequest>, ParseError> {
    let Some(headers_end) = find_header_end(buffer) else {
        return Ok(None);
    };

    let header_bytes = &buffer[..headers_end];
    let header_text = str::from_utf8(header_bytes)
        .map_err(|_| ParseError::new("invalid http header encoding"))?;
    let mut lines = header_text.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| ParseError::new("missing request line"))?;
    let (method, path, version) = parse_request_line(request_line)?;

    if version != "HTTP/1.1" {
        return Err(ParseError::new("unsupported http version"));
    }

    let mut content_length = None;
    let mut saw_transfer_encoding = false;
    let mut connection_close = false;

    for line in lines {
        if line.is_empty() {
            continue;
        }

        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| ParseError::new("invalid header line"))?;
        let header_name = name.trim();
        let header_value = value.trim();

        if header_name.eq_ignore_ascii_case("content-length") {
            let parsed = parse_content_length(header_value)?;
            match content_length {
                Some(existing) if existing != parsed => {
                    return Err(ParseError::new("conflicting content-length headers"));
                }
                _ => content_length = Some(parsed),
            }
        } else if header_name.eq_ignore_ascii_case("transfer-encoding") {
            saw_transfer_encoding = true;
        } else if header_name.eq_ignore_ascii_case("connection") {
            connection_close = header_value
                .split(',')
                .any(|token| token.trim().eq_ignore_ascii_case("close"));
        }
    }

    if saw_transfer_encoding {
        return Err(ParseError::new("transfer-encoding is not supported"));
    }

    let body_len = content_length.unwrap_or(0);
    if body_len > MAX_BODY_BYTES {
        return Err(ParseError::new("request body too large"));
    }

    let body_start = headers_end + 4;
    let total_len = body_start
        .checked_add(body_len)
        .ok_or_else(|| ParseError::new("request too large"))?;

    if buffer.len() < total_len {
        return Ok(None);
    }

    let route = match (method, path) {
        ("GET", "/ready") => Route::Ready,
        ("POST", "/fraud-score") => Route::FraudScore,
        ("GET", _) | ("POST", _) => Route::NotFound,
        _ => Route::MethodNotAllowed,
    };

    if matches!(route, Route::FraudScore) && content_length.is_none() {
        return Err(ParseError::new("missing content-length"));
    }

    Ok(Some(ParsedRequest {
        route,
        body_start,
        body_end: total_len,
        total_len,
        connection_close,
    }))
}

pub fn consume_front(buffer: &mut Vec<u8>, consumed: usize) {
    if consumed >= buffer.len() {
        buffer.clear();
        return;
    }

    buffer.copy_within(consumed.., 0);
    buffer.truncate(buffer.len() - consumed);
}

pub fn empty_response(status: u16, connection_close: bool) -> ResponseBytes {
    build_response(status, Vec::new(), connection_close)
}

pub fn json_response(status: u16, body: Vec<u8>, connection_close: bool) -> ResponseBytes {
    build_response(status, body, connection_close)
}

pub fn bad_request_response(message: &str) -> ResponseBytes {
    let escaped = message.replace('"', "\\\"");
    let body = format!(r#"{{"error":"{escaped}"}}"#).into_bytes();
    build_response(400, body, true)
}

fn build_response(status: u16, body: Vec<u8>, connection_close: bool) -> ResponseBytes {
    let reason = reason_phrase(status);
    let connection_header = if connection_close {
        "Connection: close\r\n"
    } else {
        "Connection: keep-alive\r\n"
    };

    let mut response = Vec::with_capacity(128 + body.len());
    response.extend_from_slice(format!("HTTP/1.1 {status} {reason}\r\n").as_bytes());
    response.extend_from_slice(connection_header.as_bytes());

    if body.is_empty() {
        response.extend_from_slice(b"Content-Length: 0\r\n\r\n");
    } else {
        response.extend_from_slice(b"Content-Type: application/json\r\n");
        response.extend_from_slice(format!("Content-Length: {}\r\n\r\n", body.len()).as_bytes());
        response.extend_from_slice(&body);
    }

    ResponseBytes {
        bytes: response,
        keep_alive: !connection_close,
    }
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    }
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn parse_request_line(line: &str) -> Result<(&str, &str, &str), ParseError> {
    let mut parts = line.split(' ');
    let method = parts
        .next()
        .ok_or_else(|| ParseError::new("invalid request line"))?;
    let path = parts
        .next()
        .ok_or_else(|| ParseError::new("invalid request line"))?;
    let version = parts
        .next()
        .ok_or_else(|| ParseError::new("invalid request line"))?;

    if parts.next().is_some() {
        return Err(ParseError::new("invalid request line"));
    }

    Ok((method, path, version))
}

fn parse_content_length(value: &str) -> Result<usize, ParseError> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(ParseError::new("invalid content-length"));
    }

    if value.len() > 1 && value.starts_with('0') {
        return Err(ParseError::new("invalid content-length"));
    }

    value
        .parse::<usize>()
        .map_err(|_| ParseError::new("invalid content-length"))
}
