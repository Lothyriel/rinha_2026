use std::collections::HashMap;

use serde::Deserialize;
use vicinity::hnsw::HNSWIndex;

mod engine;
mod loader;
mod math;
mod search;
mod vectorize;

const K_NEIGHBORS: usize = 5;
const FRAUD_APPROVAL_THRESHOLD: f32 = 0.6;
const VECTOR_DIMENSIONS: usize = 14;
const PADDED_VECTOR_DIMENSIONS: usize = 16;
const LEAF_SIZE: usize = 32;
const VP_NONE: u32 = u32::MAX;
const EXACT_STACK_CAPACITY: usize = 256;
const PIVOT_SAMPLE_SIZE: usize = 64;
const EPSILON: f32 = 1e-6;

#[repr(align(32))]
#[derive(Debug, Clone, Copy)]
struct Vec16([f32; PADDED_VECTOR_DIMENSIONS]);

#[derive(Debug, Clone, Copy)]
struct VpNode {
    pivot_idx: u32,
    radius: f32,
    left: u32,
    right: u32,
    start: u32,
    len: u32,
}

#[derive(Debug)]
struct ExactSearchIndex {
    nodes: Vec<VpNode>,
    indices: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
struct ExactKnnResult {
    best_distances: [f32; K_NEIGHBORS],
    best_indices: [u32; K_NEIGHBORS],
    found: usize,
}

#[derive(Debug, Clone, Copy)]
struct PointDistance {
    idx: u32,
    dist: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct HnswConfig {
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            ef_construction: 200,
            ef_search: 64,
        }
    }
}

impl HnswConfig {
    pub fn from_env() -> Self {
        let default = Self::default();

        Self {
            ef_construction: crate::read_positive_number_env(
                "RINHA_HNSW_EF_CONSTRUCTION",
                default.ef_construction,
            ),
            ef_search: crate::read_positive_number_env("RINHA_HNSW_EF_SEARCH", default.ef_search),
        }
    }
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

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum ReferenceLabel {
    Fraud,
    Legit,
}

#[derive(Debug)]
struct StoredReference {
    vector: [f32; VECTOR_DIMENSIONS],
    padded_vector: Vec16,
    is_fraud: bool,
}

#[derive(Debug)]
struct HnswSearchIndex {
    index: HNSWIndex,
    ef_search: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchBackendKind {
    Exact,
    Hnsw,
}

impl SearchBackendKind {
    pub fn from_env(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "exact" => Some(Self::Exact),
            "hnsw" => Some(Self::Hnsw),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Hnsw => "hnsw",
        }
    }
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
enum SearchIndex {
    Exact(ExactSearchIndex),
    Hnsw(HnswSearchIndex),
}

#[derive(Debug)]
pub struct FraudEngine {
    normalization: NormalizationConfig,
    mcc_risk: HashMap<String, f32>,
    references: Vec<StoredReference>,
    search_index: SearchIndex,
}
