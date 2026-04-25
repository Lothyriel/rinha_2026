use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

use chrono::{DateTime, Utc};
use flate2::read::GzDecoder;

use super::*;

pub fn load_references(resources_dir: &Path) -> Result<Vec<StoredReference>, FraudEngineError> {
    let compressed_path = resources_dir.join("references.json.gz");

    if compressed_path.exists() {
        let file = File::open(&compressed_path).map_err(|error| {
            FraudEngineError::Load(format!(
                "failed to open {}: {error}",
                compressed_path.display()
            ))
        })?;

        let reader = BufReader::new(GzDecoder::new(file));

        let raw_references: Vec<RawReferenceEntry> =
            serde_json::from_reader(reader).map_err(|error| {
                FraudEngineError::Load(format!(
                    "failed to parse {}: {error}",
                    compressed_path.display()
                ))
            })?;

        return Ok(raw_references
            .into_iter()
            .map(StoredReference::from)
            .collect());
    }

    let example_path = resources_dir.join("example-references.json");
    let raw_references: Vec<RawReferenceEntry> = load_json_file(example_path)?;

    Ok(raw_references
        .into_iter()
        .map(StoredReference::from)
        .collect())
}

pub fn load_json_file<T: serde::de::DeserializeOwned>(
    path: PathBuf,
) -> Result<T, FraudEngineError> {
    let file = File::open(&path).map_err(|error| {
        FraudEngineError::Load(format!("failed to open {}: {error}", path.display()))
    })?;

    let reader = BufReader::new(file);

    serde_json::from_reader(reader).map_err(|error| {
        FraudEngineError::Load(format!("failed to parse {}: {error}", path.display()))
    })
}

pub fn parse_utc_timestamp(value: &str) -> Result<DateTime<Utc>, FraudEngineError> {
    DateTime::parse_from_rfc3339(value)
        .map(|timestamp| timestamp.with_timezone(&Utc))
        .map_err(|error| {
            FraudEngineError::InvalidRequest(format!("invalid timestamp '{value}': {error}"))
        })
}

impl From<RawReferenceEntry> for StoredReference {
    fn from(value: RawReferenceEntry) -> Self {
        Self {
            padded_vector: Vec16::from_vector(value.vector),
            label: value.label,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn loads_official_reference_dataset_when_present() {
        let engine =
            FraudEngine::load(Path::new("./spec/resources")).expect("spec resources should load");

        assert!(engine.reference_count() >= 100_000);
    }
}
