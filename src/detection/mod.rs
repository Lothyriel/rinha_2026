use std::{collections::HashMap, path::Path};

use serde::Deserialize;

mod engine;
mod loader;
mod math;
mod search;
mod shared;
mod vectorize;

const K_NEIGHBORS: usize = 5;
const FRAUD_APPROVAL_THRESHOLD: f32 = 0.6;
const VECTOR_DIMENSIONS: usize = 14;
const PADDED_VECTOR_DIMENSIONS: usize = 16;
const LEAF_SIZE: usize = 8;
const VP_NONE: u32 = u32::MAX;
const EXACT_STACK_CAPACITY: usize = 256;
const PIVOT_SAMPLE_SIZE: usize = 64;
const QUANTIZATION_SCALE: f32 = 32_767.0;

#[repr(C, align(32))]
#[derive(Debug, Clone, Copy)]
struct QuantizedVec16([i16; PADDED_VECTOR_DIMENSIONS]);

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct VpNode {
    pivot_idx: u32,
    radius: u64,
    left: u32,
    right: u32,
    start: u32,
    len: u32,
}

#[derive(Debug, Clone)]
struct ExactSearchIndex {
    nodes: Vec<VpNode>,
    indices: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
struct ExactKnnResult {
    best_distances: [u64; K_NEIGHBORS],
    best_indices: [u32; K_NEIGHBORS],
    found: usize,
}

#[derive(Debug, Clone, Copy)]
struct PointDistance {
    idx: u32,
    dist: u64,
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
    padded_vector: QuantizedVec16,
    label: ReferenceLabel,
}

#[derive(Debug)]
struct OwnedDataset {
    references: Vec<StoredReference>,
    index: ExactSearchIndex,
}

#[derive(Debug)]
enum DatasetStorage {
    Owned(OwnedDataset),
    Shared(shared::MappedDataset),
}

#[derive(Debug)]
pub struct FraudEngine {
    normalization: NormalizationConfig,
    mcc_risk: HashMap<String, f32>,
    dataset: DatasetStorage,
}

pub fn prebuild_shared_dataset(
    resources_dir: &Path,
    mmap_path: &Path,
) -> Result<(), FraudEngineError> {
    shared::build_shared_dataset_file(resources_dir, mmap_path, LEAF_SIZE)
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
            Self::Owned(dataset) => dataset.references.len(),
            Self::Shared(dataset) => dataset.len(),
        }
    }

    fn vector(&self, index: usize) -> &QuantizedVec16 {
        match self {
            Self::Owned(dataset) => &dataset.references[index].padded_vector,
            Self::Shared(dataset) => dataset.vector(index),
        }
    }

    fn label(&self, index: usize) -> ReferenceLabel {
        match self {
            Self::Owned(dataset) => dataset.references[index].label,
            Self::Shared(dataset) => dataset.label(index),
        }
    }

    fn nodes(&self) -> &[VpNode] {
        match self {
            Self::Owned(dataset) => &dataset.index.nodes,
            Self::Shared(dataset) => dataset.nodes(),
        }
    }

    fn indices(&self) -> &[u32] {
        match self {
            Self::Owned(dataset) => &dataset.index.indices,
            Self::Shared(dataset) => dataset.indices(),
        }
    }
}
