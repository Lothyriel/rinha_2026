use std::collections::HashMap;

use serde::Deserialize;

mod engine;
mod loader;
mod math;
mod search;
pub mod simd;
mod topk;
mod vectorize;

const K_NEIGHBORS: usize = 5;
const FRAUD_APPROVAL_THRESHOLD: f32 = 0.6;
const VECTOR_DIMENSIONS: usize = 14;
const IVF_CLUSTER_COUNT: usize = 2_048;
const BLOCK_WIDTH: usize = 8;
const QUANTIZATION_SCALE: f32 = 10_000.0;
const PADDED_LABEL_VALUE: u8 = u8::MAX;
const PREBUILT_INDEX_MAGIC: [u8; 4] = *b"IVF2";

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
    vector: [f32; VECTOR_DIMENSIONS],
    label: ReferenceLabel,
}

#[repr(u8)]
#[derive(Debug, Deserialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum ReferenceLabel {
    Fraud,
    Legit,
}

#[derive(Debug, Clone, Copy)]
struct StoredReference {
    vector: [f32; VECTOR_DIMENSIONS],
    label: ReferenceLabel,
}

#[derive(Debug, Clone)]
struct IvfIndex {
    reference_count: usize,
    centroids_transposed: Vec<f32>,
    radii: Vec<f32>,
    block_offsets: Vec<u32>,
    labels: Vec<u8>,
    quantized_blocks: Vec<i16>,
}

#[derive(Debug)]
struct OwnedDataset {
    index: IvfIndex,
}

#[derive(Debug, Clone, Copy)]
struct SearchConfig {
    nprobe: Option<usize>,
    fast_nprobe: Option<usize>,
}

#[derive(Debug)]
pub struct FraudEngine {
    normalization: NormalizationConfig,
    mcc_risk: HashMap<String, f32>,
    dataset: OwnedDataset,
    search: SearchConfig,
}

impl ReferenceLabel {
    fn to_storage_byte(self) -> u8 {
        self as u8
    }

    fn is_fraud_storage_byte(value: u8) -> bool {
        value == Self::Fraud as u8
    }
}

impl OwnedDataset {
    fn len(&self) -> usize {
        self.index.reference_count
    }

    fn cluster_count(&self) -> usize {
        self.index.cluster_count()
    }

    fn centroid_component(&self, dimension: usize, cluster: usize) -> f32 {
        self.index.centroid_component(dimension, cluster)
    }

    fn radius(&self, cluster: usize) -> f32 {
        self.index.radii[cluster]
    }

    fn block_offsets(&self) -> &[u32] {
        &self.index.block_offsets
    }

    fn labels(&self) -> &[u8] {
        &self.index.labels
    }

    fn quantized_blocks(&self) -> &[i16] {
        &self.index.quantized_blocks
    }
}

impl IvfIndex {
    fn cluster_count(&self) -> usize {
        self.radii.len()
    }

    fn centroid_component(&self, dimension: usize, cluster: usize) -> f32 {
        self.centroids_transposed[dimension * self.cluster_count() + cluster]
    }
}
