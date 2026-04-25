use std::{fs, net::SocketAddr, path::Path, sync::Arc};

use rinha_2026::{app, detection};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main(flavor = "current_thread")]
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

    let engine = Arc::new(detection::FraudEngine::load(resources_dir.as_ref())?);

    let reference_count = engine.reference_count();
    let app = app::router(app::AppState { engine });

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

        tracing::info!(
            socket_path = %socket_path.display(),
            reference_count,
            resources_dir,
            "fraud API listening on unix socket"
        );

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await?;

        return Ok(());
    }

    let address = SocketAddr::from(([0, 0, 0, 0], 9999));
    let listener = tokio::net::TcpListener::bind(address).await?;

    tracing::info!(
        %address,
        reference_count,
        resources_dir,
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

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("shutdown signal received");
}
