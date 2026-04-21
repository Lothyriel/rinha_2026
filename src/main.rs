use std::{fs, net::SocketAddr, path::Path, sync::Arc};

use rinha_2026::{app, detection};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rinha_2026=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let resources_dir =
        std::env::var("RINHA_RESOURCES_DIR").unwrap_or_else(|_| "./spec/resources".to_owned());

    let port = std::env::var("PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(9999);

    let configured_search_backend = std::env::var("RINHA_SEARCH_BACKEND")
        .ok()
        .and_then(|value| detection::SearchBackendKind::from_env(&value))
        .unwrap_or(detection::SearchBackendKind::Exact);

    let engine = Arc::new(detection::FraudEngine::load(
        resources_dir.as_ref(),
        configured_search_backend,
    )?);

    let reference_count = engine.reference_count();
    let active_search_backend = engine.search_backend_name();
    let app = app::router(app::AppState { engine });

    #[cfg(unix)]
    if let Some(socket_path) = std::env::var("RINHA_UNIX_SOCKET_PATH")
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        let socket_path = Path::new(socket_path.trim());

        if let Some(parent) = socket_path.parent() {
            fs::create_dir_all(parent)?;
        }

        match fs::remove_file(socket_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }

        let listener = tokio::net::UnixListener::bind(socket_path)?;
        fs::set_permissions(socket_path, fs::Permissions::from_mode(0o666))?;

        tracing::info!(
            socket_path = %socket_path.display(),
            reference_count,
            resources_dir,
            configured_search_backend = configured_search_backend.as_str(),
            active_search_backend,
            "fraud API listening on unix socket"
        );

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await?;

        return Ok(());
    }

    let address = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(address).await?;

    tracing::info!(
        %address,
        reference_count,
        resources_dir,
        configured_search_backend = configured_search_backend.as_str(),
        active_search_backend,
        "fraud API listening on tcp"
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            tracing::warn!(?error, "failed to install CTRL+C handler");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(error) => {
                tracing::warn!(?error, "failed to install SIGTERM handler");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("shutdown signal received");
}
