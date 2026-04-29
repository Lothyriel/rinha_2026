use std::{
    io,
    net::SocketAddr,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use monoio::{
    io::{AsyncReadRent, AsyncWriteRent, AsyncWriteRentExt},
    net::{TcpListener, UnixListener},
};

use crate::detection::FraudEngine;

mod http1;
mod json;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<FraudEngine>,
}

pub async fn run_tcp(
    address: SocketAddr,
    state: AppState,
    shutdown: Arc<AtomicBool>,
) -> io::Result<()> {
    let listener = TcpListener::bind(address)?;

    loop {
        if shutdown.load(Ordering::Acquire) {
            return Ok(());
        }

        let (stream, _) = listener.accept().await?;
        if shutdown.load(Ordering::Acquire) {
            return Ok(());
        }

        let state = state.clone();
        let shutdown = shutdown.clone();
        monoio::spawn(async move {
            if let Err(error) = serve_connection(stream, state, shutdown).await {
                tracing::debug!(?error, "connection ended with error");
            }
        });
    }
}

pub async fn run_unix(
    socket_path: &Path,
    state: AppState,
    shutdown: Arc<AtomicBool>,
) -> io::Result<()> {
    let listener = UnixListener::bind(socket_path)?;

    let _ = std::fs::set_permissions(socket_path, std::fs::Permissions::from_mode(0o666));

    loop {
        if shutdown.load(Ordering::Acquire) {
            return Ok(());
        }

        let (stream, _) = listener.accept().await?;
        if shutdown.load(Ordering::Acquire) {
            return Ok(());
        }

        let state = state.clone();
        let shutdown = shutdown.clone();
        monoio::spawn(async move {
            if let Err(error) = serve_connection(stream, state, shutdown).await {
                tracing::debug!(?error, "connection ended with error");
            }
        });
    }
}

pub async fn serve_connection<S>(
    mut stream: S,
    state: AppState,
    shutdown: Arc<AtomicBool>,
) -> io::Result<()>
where
    S: AsyncReadRent + AsyncWriteRent + Unpin,
{
    const READ_CAPACITY: usize = 8 * 1024;

    let mut pending = Vec::with_capacity(READ_CAPACITY);
    let mut read_buf = Vec::with_capacity(READ_CAPACITY);

    loop {
        match http1::try_parse_request(&pending) {
            Ok(Some(request)) => {
                let response = match request.route {
                    http1::Route::Ready => http1::empty_response(200, request.connection_close),
                    http1::Route::FraudScore => {
                        let body = &pending[request.body_start..request.body_end];
                        json::handle_fraud_score(body, &state.engine, request.connection_close)
                    }
                    http1::Route::NotFound => http1::json_response(
                        404,
                        br#"{"error":"not found"}"#.to_vec(),
                        request.connection_close,
                    ),
                    http1::Route::MethodNotAllowed => http1::json_response(
                        405,
                        br#"{"error":"method not allowed"}"#.to_vec(),
                        request.connection_close,
                    ),
                };

                let keep_alive = response.keep_alive && !shutdown.load(Ordering::Acquire);
                write_response(&mut stream, response.bytes).await?;
                http1::consume_front(&mut pending, request.total_len);

                if !keep_alive {
                    let _ = stream.shutdown().await;
                    return Ok(());
                }

                continue;
            }
            Ok(None) => {}
            Err(error) => {
                let response = http1::bad_request_response(error.message());
                write_response(&mut stream, response.bytes).await?;
                let _ = stream.shutdown().await;
                return Ok(());
            }
        }

        if shutdown.load(Ordering::Acquire) {
            let _ = stream.shutdown().await;
            return Ok(());
        }

        let (result, buffer) = stream.read(read_buf).await;
        read_buf = buffer;
        let read = result?;

        if read == 0 {
            let _ = stream.shutdown().await;
            return Ok(());
        }

        pending.extend_from_slice(&read_buf[..read]);
        read_buf.clear();

        if pending.len() > http1::MAX_BUFFERED_BYTES {
            let response = http1::bad_request_response("request too large");
            write_response(&mut stream, response.bytes).await?;
            let _ = stream.shutdown().await;
            return Ok(());
        }
    }
}

async fn write_response<S>(stream: &mut S, response: Vec<u8>) -> io::Result<()>
where
    S: AsyncWriteRent + Unpin,
{
    let (result, _) = stream.write_all(response).await;
    result.map(|_| ())
}

#[derive(Clone, Debug)]
pub enum WakeTarget {
    Tcp(SocketAddr),
    Unix(PathBuf),
}

pub fn wake_accept(target: &WakeTarget) {
    match target {
        WakeTarget::Tcp(address) => {
            let _ = std::net::TcpStream::connect(address);
        }
        WakeTarget::Unix(path) => {
            let _ = std::os::unix::net::UnixStream::connect(path);
        }
    }
}
