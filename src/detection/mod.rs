use std::{collections::HashMap, path::Path};

use serde::Deserialize;

mod clustering;
mod engine;
mod loader;
mod math;
mod search;
mod shared;
mod vectorize;

const K_NEIGHBORS: usize = 5;
const FRAUD_APPROVAL_THRESHOLD: f32 = 0.6;
const VECTOR_DIMENSIONS: usize = 14;
const STORED_VECTOR_DIMENSIONS: usize = 16;
const QUANTIZATION_SCALE: f32 = 16_383.0;
const DEFAULT_KMEANS_SEED: u64 = 67;

#[repr(C, align(32))]
#[derive(Debug, Clone, Copy)]
struct QuantizedVector([i16; STORED_VECTOR_DIMENSIONS]);

#[derive(Debug, Clone, Copy)]
struct ExactKnnResult {
    best_distances: [i64; K_NEIGHBORS],
    best_indices: [u32; K_NEIGHBORS],
    found: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum FraudEngineError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("engine unavailable: {0}")]
    Unavailable(String),
    #[error("engine load failed: {0}")]
    Load(String),
}

#[derive(Debug, Deserialize)]
pub struct NormalizationConfig {
    pub max_amount: f32,
    pub max_installments: f32,
    pub amount_vs_avg_ratio: f32,
    pub max_minutes: f32,
    pub max_km: f32,
    pub max_tx_count_24h: f32,
    pub max_merchant_avg_amount: f32,
}

#[derive(Debug, Deserialize)]
struct RawReferenceEntry {
    vector: [f32; 14],
    label: ReferenceLabel,
}

#[repr(u8)]
#[derive(Debug, Deserialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum ReferenceLabel {
    Fraud,
    Legit,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct StoredReference {
    quantized_vector: QuantizedVector,
    label: ReferenceLabel,
}

#[derive(Debug)]
struct OwnedDataset {
    vectors: Vec<QuantizedVector>,
    labels: Vec<u8>,
}

#[derive(Debug)]
enum DatasetStorage {
    Owned(OwnedDataset),
    Embedded(shared::EmbeddedDataset),
}

#[derive(Debug)]
pub struct FraudEngine {
    normalization: NormalizationConfig,
    mcc_risk: HashMap<String, f32>,
    dataset: DatasetStorage,
}

#[derive(Debug, Clone, Copy)]
pub struct DatasetBuildOptions {
    kmeans_k: usize,
    kmeans_seed: u64,
}

pub fn prebuild_shared_dataset(
    resources_dir: &Path,
    mmap_path: &Path,
) -> Result<(), FraudEngineError> {
    prebuild_shared_dataset_with_options(resources_dir, mmap_path, DatasetBuildOptions::from_env()?)
}

pub fn prebuild_shared_dataset_with_options(
    resources_dir: &Path,
    mmap_path: &Path,
    options: DatasetBuildOptions,
) -> Result<(), FraudEngineError> {
    clustering::build_dataset_file(resources_dir, mmap_path, options)
}

impl DatasetBuildOptions {
    pub fn exact() -> Self {
        Self {
            kmeans_k: 0,
            kmeans_seed: DEFAULT_KMEANS_SEED,
        }
    }

    pub fn fixed_clustered(kmeans_k: usize, kmeans_seed: u64) -> Self {
        Self {
            kmeans_k,
            kmeans_seed,
        }
    }

    pub fn from_env() -> Result<Self, FraudEngineError> {
        let seed = parse_env_u64("RINHA_KMEANS_SEED", DEFAULT_KMEANS_SEED)?;

        match std::env::var("RINHA_KMEANS_K") {
            Ok(value) => {
                let parsed = parse_positive_usize("RINHA_KMEANS_K", value.trim())?;
                if parsed == 0 {
                    Ok(Self::exact())
                } else {
                    Ok(Self::fixed_clustered(parsed, seed))
                }
            }
            Err(std::env::VarError::NotPresent) => Ok(Self {
                kmeans_seed: seed,
                ..Self::exact()
            }),
            Err(error) => Err(FraudEngineError::Load(format!(
                "failed to read RINHA_KMEANS_K: {error}"
            ))),
        }
    }

    fn configured_k(&self) -> usize {
        self.kmeans_k
    }

    fn seed(&self) -> u64 {
        self.kmeans_seed
    }

    fn effective_k(&self, source_count: usize) -> usize {
        if source_count == 0 {
            return 0;
        }

        self.kmeans_k.min(source_count)
    }

    fn clustering_enabled_for(&self, source_count: usize) -> bool {
        let effective_k = self.effective_k(source_count);
        effective_k >= K_NEIGHBORS && effective_k < source_count
    }

    pub fn clustering_enabled(&self, source_count: usize) -> bool {
        self.clustering_enabled_for(source_count)
    }

    pub fn effective_reference_count(&self, source_count: usize) -> usize {
        self.effective_k(source_count)
    }

    pub fn configured_kmeans_k(&self) -> usize {
        self.configured_k()
    }

    pub fn kmeans_seed(&self) -> u64 {
        self.seed()
    }
}

impl ReferenceLabel {
    fn from_storage_byte(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Fraud),
            1 => Some(Self::Legit),
            _ => None,
        }
    }

    fn to_storage_byte(self) -> u8 {
        self as u8
    }
}

impl DatasetStorage {
    fn len(&self) -> usize {
        match self {
            Self::Owned(dataset) => dataset.vectors.len(),
            Self::Embedded(dataset) => dataset.len(),
        }
    }

    fn vector(&self, index: usize) -> &QuantizedVector {
        match self {
            Self::Owned(dataset) => &dataset.vectors[index],
            Self::Embedded(dataset) => dataset.vector(index),
        }
    }

    fn label(&self, index: usize) -> ReferenceLabel {
        match self {
            Self::Owned(dataset) => ReferenceLabel::from_storage_byte(dataset.labels[index])
                .expect("owned dataset stores valid labels"),
            Self::Embedded(dataset) => dataset.label(index),
        }
    }
}

fn parse_env_u64(name: &str, default: u64) -> Result<u64, FraudEngineError> {
    match std::env::var(name) {
        Ok(value) => value.trim().parse::<u64>().map_err(|error| {
            FraudEngineError::Load(format!("invalid {name} value '{}': {error}", value.trim()))
        }),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(FraudEngineError::Load(format!(
            "failed to read {name}: {error}"
        ))),
    }
}

fn parse_positive_usize(name: &str, value: &str) -> Result<usize, FraudEngineError> {
    value
        .parse::<usize>()
        .map_err(|error| FraudEngineError::Load(format!("invalid {name} value '{value}': {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_kmeans_respects_requested_k() {
        let options = DatasetBuildOptions::fixed_clustered(17, 42);
        assert_eq!(options.effective_k(100), 17);
        assert_eq!(options.effective_k(8), 8);
    }
}
