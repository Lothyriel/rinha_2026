use std::{env, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rinha_2026=info".into()),
        )
        .init();

    let mut args = env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "prebuild_shared_dataset".to_owned());
    let resources_dir = args.next();
    let output_path = args.next();
    let kmeans_k = args.next();

    if args.next().is_some() || resources_dir.is_none() || output_path.is_none() {
        return Err(
            format!("usage: {program} <resources-dir> <output-mmap-path> [kmeans-k]").into(),
        );
    }

    let resources_dir = resources_dir.expect("resources_dir checked above");
    let output_path = output_path.expect("output_path checked above");

    tracing::info!(
        resources_dir,
        output_path,
        requested_kmeans_k = kmeans_k.as_deref().unwrap_or("env/default"),
        "prebuilding shared dataset mmap artifact"
    );

    let options = match kmeans_k {
        Some(value) => {
            let parsed = value.parse::<usize>()?;
            if parsed == 0 {
                rinha_2026::detection::DatasetBuildOptions::exact()
            } else {
                rinha_2026::detection::DatasetBuildOptions::fixed_clustered(parsed, 67)
            }
        }
        None => rinha_2026::detection::DatasetBuildOptions::from_env()?,
    };

    tracing::info!(
        configured_kmeans_k = options.configured_kmeans_k(),
        seed = options.kmeans_seed(),
        "resolved prebuild dataset options"
    );

    rinha_2026::detection::prebuild_shared_dataset_with_options(
        Path::new(&resources_dir),
        Path::new(&output_path),
        options,
    )?;

    tracing::info!(output_path, "shared dataset mmap artifact ready");

    Ok(())
}
