use std::{
    fs,
    net::SocketAddr,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use rinha_2026::{detection, server};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[monoio::main(driver = "fusion")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rinha_2026=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let resources_dir =
        std::env::var("RINHA_RESOURCES_DIR").unwrap_or_else(|_| "./spec/resources".to_owned());

    let engine = Arc::new(detection::FraudEngine::load(resources_dir.as_ref())?);

    let reference_count = engine.reference_count();
    let state = server::AppState { engine };

    if let Some(socket_path) = std::env::var("RINHA_UNIX_SOCKET_PATH")
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        let socket_path = PathBuf::from(socket_path.trim());

        if let Some(parent) = socket_path.parent() {
            fs::create_dir_all(parent)?;
        }

        match fs::remove_file(&socket_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }

        let _ = std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o666));
        let shutdown = install_shutdown_handler(server::WakeTarget::Unix(socket_path.clone()))?;

        tracing::info!(
            socket_path = %socket_path.display(),
            reference_count,
            resources_dir,
            "fraud API listening on unix socket"
        );

        let result = server::run_unix(Path::new(&socket_path), state, shutdown).await;
        let _ = fs::remove_file(&socket_path);
        result?;
        return Ok(());
    }

    let port = std::env::var("PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(9999);
    let address = SocketAddr::from(([0, 0, 0, 0], port));
    let shutdown = install_shutdown_handler(server::WakeTarget::Tcp(address))?;

    tracing::info!(
        %address,
        reference_count,
        resources_dir,
        "fraud API listening on tcp"
    );

    server::run_tcp(address, state, shutdown).await?;

    Ok(())
}

fn install_shutdown_handler(
    wake_target: server::WakeTarget,
) -> Result<Arc<AtomicBool>, Box<dyn std::error::Error>> {
    let shutdown = Arc::new(AtomicBool::new(false));
    let handler_state = shutdown.clone();

    ctrlc::set_handler(move || {
        if !handler_state.swap(true, Ordering::AcqRel) {
            tracing::info!("shutdown signal received");
            server::wake_accept(&wake_target);
        }
    })?;

    Ok(shutdown)
}
