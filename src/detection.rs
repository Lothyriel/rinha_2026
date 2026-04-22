use std::{
    collections::HashMap,
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    time::Instant,
};

use chrono::{DateTime, Datelike, Timelike, Utc};
use flate2::read::GzDecoder;
use metrics::histogram;
use serde::Deserialize;
use vicinity::hnsw::HNSWIndex;

use crate::model::*;

const K_NEIGHBORS: usize = 5;
const FRAUD_APPROVAL_THRESHOLD: f32 = 0.6;

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
    vector: [f32; 14],
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
    Exact,
    Hnsw(HnswSearchIndex),
}

#[derive(Debug)]
pub struct FraudEngine {
    normalization: NormalizationConfig,
    mcc_risk: HashMap<String, f32>,
    references: Vec<StoredReference>,
    search_index: SearchIndex,
}

impl FraudEngine {
    pub fn load(
        resources_dir: &Path,
        configured_search_backend: SearchBackendKind,
        hnsw_config: HnswConfig,
    ) -> Result<Self, FraudEngineError> {
        let normalization = load_json_file(resources_dir.join("normalization.json"))?;
        let mcc_risk = load_json_file(resources_dir.join("mcc_risk.json"))?;
        let references = load_references(resources_dir)?;

        if references.len() < K_NEIGHBORS {
            return Err(FraudEngineError::Load(format!(
                "reference dataset must contain at least {K_NEIGHBORS} vectors, found {}",
                references.len()
            )));
        }

        let search_index = build_search_index(&references, configured_search_backend, hnsw_config);

        Ok(Self {
            normalization,
            mcc_risk,
            references,
            search_index,
        })
    }

    pub fn score(
        &self,
        request: &FraudScoreRequest,
    ) -> Result<FraudScoreResponse, FraudEngineError> {
        let vectorize_start = Instant::now();

        let vector = self.vectorize(request)?;

        histogram!("score_engine", "step" => "vectorize")
            .record(vectorize_start.elapsed().as_micros() as f64);

        let classify_start = Instant::now();

        let (neighbor_count, fraud_votes) = self.classify_knn(&vector, K_NEIGHBORS);

        histogram!(
            "score_engine",
            "step" => "classify"
        )
        .record(classify_start.elapsed().as_micros() as f64);

        if neighbor_count == 0 {
            return Err(FraudEngineError::Unavailable(
                "reference dataset is empty".to_owned(),
            ));
        }

        let fraud_score = fraud_votes as f32 / neighbor_count as f32;

        Ok(FraudScoreResponse {
            approved: fraud_score < FRAUD_APPROVAL_THRESHOLD,
            fraud_score,
        })
    }

    pub fn reference_count(&self) -> usize {
        self.references.len()
    }

    pub fn search_backend_name(&self) -> &'static str {
        match &self.search_index {
            SearchIndex::Exact => "exact",
            SearchIndex::Hnsw(_) => "hnsw",
        }
    }

    fn vectorize(&self, req: &FraudScoreRequest) -> Result<[f32; 14], FraudEngineError> {
        let requested_at = parse_utc_timestamp(&req.transaction.requested_at)?;

        let amount = clamp_ratio(req.transaction.amount as f32, self.normalization.max_amount);

        let installments = clamp_ratio(
            req.transaction.installments as f32,
            self.normalization.max_installments,
        );

        let amount_vs_avg = normalize_amount_vs_avg(
            req.transaction.amount as f32,
            req.customer.avg_amount as f32,
            self.normalization.amount_vs_avg_ratio,
        );

        let hour_of_day = requested_at.hour() as f32 / 23.0;
        let day_of_week = requested_at.weekday().num_days_from_monday() as f32 / 6.0;

        let (minutes_since_last_tx, km_from_last_tx) = self.get_last_tx_data(req, requested_at)?;

        let km_from_home = clamp_ratio(req.terminal.km_from_home as f32, self.normalization.max_km);
        let tx_count_24h = clamp_ratio(
            req.customer.tx_count_24h as f32,
            self.normalization.max_tx_count_24h,
        );

        let is_online = bool_to_unit(req.terminal.is_online);
        let card_present = bool_to_unit(req.terminal.card_present);

        let unknown_merchant = if req
            .customer
            .known_merchants
            .iter()
            .any(|known_merchant| known_merchant == &req.merchant.id)
        {
            0.0
        } else {
            1.0
        };

        let mcc_risk = self.mcc_risk.get(&req.merchant.mcc).copied().unwrap_or(0.5);

        let merchant_avg_amount = clamp_ratio(
            req.merchant.avg_amount as f32,
            self.normalization.max_merchant_avg_amount,
        );

        Ok([
            amount,
            installments,
            amount_vs_avg,
            hour_of_day,
            day_of_week,
            minutes_since_last_tx,
            km_from_last_tx,
            km_from_home,
            tx_count_24h,
            is_online,
            card_present,
            unknown_merchant,
            mcc_risk,
            merchant_avg_amount,
        ])
    }

    fn get_last_tx_data(
        &self,
        req: &FraudScoreRequest,
        requested_at: DateTime<Utc>,
    ) -> Result<(f32, f32), FraudEngineError> {
        let Some(last_transaction) = &req.last_transaction else {
            return Ok((-1.0, -1.0));
        };

        let last_timestamp = parse_utc_timestamp(&last_transaction.timestamp)?;

        let elapsed_seconds = requested_at
            .signed_duration_since(last_timestamp)
            .num_seconds()
            .max(0) as f32;

        let time = clamp_ratio(elapsed_seconds / 60.0, self.normalization.max_minutes);

        let distance = clamp_ratio(
            last_transaction.km_from_current as f32,
            self.normalization.max_km,
        );

        Ok((time, distance))
    }

    fn classify_knn(&self, query: &[f32; 14], neighbors: usize) -> (usize, usize) {
        let start = Instant::now();

        let result = match &self.search_index {
            SearchIndex::Exact => self.classify_exact(query, neighbors),
            SearchIndex::Hnsw(index) => self.classify_hnsw(query, neighbors, index),
        };

        histogram!(
            "score_engine",
            "step" => "classify"
        )
        .record(start.elapsed().as_micros() as f64);

        result
    }

    fn classify_exact(&self, query: &[f32; 14], neighbors: usize) -> (usize, usize) {
        let mut best_distances = [f32::INFINITY; K_NEIGHBORS];
        let mut best_is_fraud = [false; K_NEIGHBORS];
        let mut found = 0;

        for reference in &self.references {
            let distance = innr::l2_distance_squared(query, &reference.vector);
            let insert_at = best_distances.partition_point(|current| *current < distance);

            if insert_at >= neighbors {
                continue;
            }

            let upper_bound = found.min(neighbors.saturating_sub(1));

            for index in (insert_at..upper_bound).rev() {
                best_distances[index + 1] = best_distances[index];
                best_is_fraud[index + 1] = best_is_fraud[index];
            }

            best_distances[insert_at] = distance;
            best_is_fraud[insert_at] = reference.is_fraud;
            found = (found + 1).min(neighbors);
        }

        let fraud_votes = best_is_fraud
            .iter()
            .zip(best_distances.iter())
            .take(found)
            .filter(|(is_fraud, distance)| **is_fraud && distance.is_finite())
            .count();

        (found, fraud_votes)
    }

    fn classify_hnsw(
        &self,
        query: &[f32; 14],
        neighbors: usize,
        index: &HnswSearchIndex,
    ) -> (usize, usize) {
        let normalized_query = normalize_l2(query);

        let results = index
            .index
            .search(&normalized_query, neighbors, index.ef_search)
            .expect("hnsw failed");

        let fraud_votes = results
            .iter()
            .filter(|(doc_id, _)| {
                self.references
                    .get(*doc_id as usize)
                    .is_some_and(|reference| reference.is_fraud)
            })
            .count();

        (results.len(), fraud_votes)
    }
}

fn build_search_index(
    references: &[StoredReference],
    configured_search_backend: SearchBackendKind,
    hnsw_config: HnswConfig,
) -> SearchIndex {
    match configured_search_backend {
        SearchBackendKind::Exact => SearchIndex::Exact,
        SearchBackendKind::Hnsw => match build_hnsw_index(references, hnsw_config) {
            Ok(index) => SearchIndex::Hnsw(index),
            Err(error) => {
                tracing::error!(
                    ?error,
                    "failed to build HNSW index; falling back to exact search"
                );
                SearchIndex::Exact
            }
        },
    }
}

fn build_hnsw_index(
    references: &[StoredReference],
    hnsw_config: HnswConfig,
) -> Result<HnswSearchIndex, FraudEngineError> {
    const VECTOR_DIMENSIONS: usize = 14;
    const HNSW_M: usize = 16;
    const HNSW_M_MAX: usize = 32;
    let build_started_at = Instant::now();

    tracing::info!(
        reference_count = references.len(),
        ef_construction = hnsw_config.ef_construction,
        ef_search = hnsw_config.ef_search,
        "building HNSW index"
    );

    let mut index = HNSWIndex::builder(VECTOR_DIMENSIONS)
        .m(HNSW_M)
        .m_max(HNSW_M_MAX)
        .ef_construction(hnsw_config.ef_construction)
        .ef_search(hnsw_config.ef_search)
        .build()
        .map_err(|error| {
            FraudEngineError::Load(format!("failed to initialize HNSW index: {error}"))
        })?;

    for (doc_id, reference) in references.iter().enumerate() {
        let normalized_vector = normalize_l2(&reference.vector);

        index
            .add_slice(doc_id as u32, &normalized_vector)
            .map_err(|error| {
                FraudEngineError::Load(format!("failed to insert HNSW vector {doc_id}: {error}"))
            })?;
    }

    index
        .build()
        .map_err(|error| FraudEngineError::Load(format!("failed to build HNSW graph: {error}")))?;

    tracing::info!(
        reference_count = references.len(),
        ef_construction = hnsw_config.ef_construction,
        ef_search = hnsw_config.ef_search,
        build_ms = build_started_at.elapsed().as_secs_f64() * 1_000.0,
        "built HNSW index"
    );

    Ok(HnswSearchIndex {
        index,
        ef_search: hnsw_config.ef_search,
    })
}

fn load_references(resources_dir: &Path) -> Result<Vec<StoredReference>, FraudEngineError> {
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

impl From<RawReferenceEntry> for StoredReference {
    fn from(value: RawReferenceEntry) -> Self {
        Self {
            vector: value.vector,
            is_fraud: value.label == ReferenceLabel::Fraud,
        }
    }
}

fn load_json_file<T: serde::de::DeserializeOwned>(path: PathBuf) -> Result<T, FraudEngineError> {
    let file = File::open(&path).map_err(|error| {
        FraudEngineError::Load(format!("failed to open {}: {error}", path.display()))
    })?;

    let reader = BufReader::new(file);

    serde_json::from_reader(reader).map_err(|error| {
        FraudEngineError::Load(format!("failed to parse {}: {error}", path.display()))
    })
}

fn parse_utc_timestamp(value: &str) -> Result<DateTime<Utc>, FraudEngineError> {
    DateTime::parse_from_rfc3339(value)
        .map(|timestamp| timestamp.with_timezone(&Utc))
        .map_err(|error| {
            FraudEngineError::InvalidRequest(format!("invalid timestamp '{value}': {error}"))
        })
}

#[inline]
fn normalize_amount_vs_avg(amount: f32, customer_average: f32, scaling_ratio: f32) -> f32 {
    if customer_average <= 0.0 {
        return if amount <= 0.0 { 0.0 } else { 1.0 };
    }

    clamp_unit((amount / customer_average) / scaling_ratio)
}

#[inline]
fn clamp_ratio(value: f32, max_value: f32) -> f32 {
    if max_value <= 0.0 {
        return 0.0;
    }

    clamp_unit(value / max_value)
}

#[inline]
fn clamp_unit(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

#[inline]
fn bool_to_unit(value: bool) -> f32 {
    if value { 1.0 } else { 0.0 }
}

#[inline]
fn normalize_l2(vector: &[f32; 14]) -> [f32; 14] {
    let mut squared_norm = 0.0;

    for value in vector {
        squared_norm += value * value;
    }

    if squared_norm <= f32::EPSILON {
        return *vector;
    }

    let norm = squared_norm.sqrt();
    let mut normalized = [0.0; 14];

    for index in 0..14 {
        normalized[index] = vector[index] / norm;
    }

    normalized
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Customer, LastTransaction, Merchant, Terminal, Transaction};

    fn engine() -> FraudEngine {
        FraudEngine::load(
            Path::new("./spec/resources"),
            SearchBackendKind::Exact,
            HnswConfig::default(),
        )
        .expect("spec resources should load")
    }

    #[test]
    fn vectorizes_missing_last_transaction_with_sentinel_values() {
        let request = FraudScoreRequest {
            id: "tx-1329056812".to_owned(),
            transaction: Transaction {
                amount: 41.12,
                installments: 2,
                requested_at: "2026-03-11T18:45:53Z".to_owned(),
            },
            customer: Customer {
                avg_amount: 82.24,
                tx_count_24h: 3,
                known_merchants: vec!["MERC-003".to_owned(), "MERC-016".to_owned()],
            },
            merchant: Merchant {
                id: "MERC-016".to_owned(),
                mcc: "5411".to_owned(),
                avg_amount: 60.25,
            },
            terminal: Terminal {
                is_online: false,
                card_present: true,
                km_from_home: 29.23,
            },
            last_transaction: None,
        };

        let vector = engine()
            .vectorize(&request)
            .expect("vector should be produced");

        assert_eq!(vector[5], -1.0);
        assert_eq!(vector[6], -1.0);
        assert_eq!(vector[9], 0.0);
        assert_eq!(vector[10], 1.0);
        assert_eq!(vector[11], 0.0);
        assert!((vector[12] - 0.15).abs() < 0.0001);
    }

    #[test]
    fn vectorizes_known_fraud_shape_with_previous_transaction() {
        let request = FraudScoreRequest {
            id: "tx-1788243118".to_owned(),
            transaction: Transaction {
                amount: 4368.82,
                installments: 8,
                requested_at: "2026-03-17T02:04:06Z".to_owned(),
            },
            customer: Customer {
                avg_amount: 68.88,
                tx_count_24h: 18,
                known_merchants: vec![
                    "MERC-004".to_owned(),
                    "MERC-015".to_owned(),
                    "MERC-017".to_owned(),
                    "MERC-007".to_owned(),
                ],
            },
            merchant: Merchant {
                id: "MERC-062".to_owned(),
                mcc: "7801".to_owned(),
                avg_amount: 25.55,
            },
            terminal: Terminal {
                is_online: true,
                card_present: false,
                km_from_home: 881.61,
            },
            last_transaction: Some(LastTransaction {
                timestamp: "2026-03-17T01:58:06Z".to_owned(),
                km_from_current: 660.92,
            }),
        };

        let vector = engine()
            .vectorize(&request)
            .expect("vector should be produced");

        assert!((vector[0] - 0.4369).abs() < 0.001);
        assert!((vector[1] - 0.6667).abs() < 0.001);
        assert_eq!(vector[9], 1.0);
        assert_eq!(vector[10], 0.0);
        assert_eq!(vector[11], 1.0);
        assert!((vector[12] - 0.8).abs() < 0.001);
    }

    #[test]
    fn loads_official_reference_dataset_when_present() {
        let engine = engine();

        assert!(engine.reference_count() >= 100_000);
    }

    #[test]
    fn parses_search_backend_names() {
        assert_eq!(
            SearchBackendKind::from_env("exact"),
            Some(SearchBackendKind::Exact)
        );
        assert_eq!(
            SearchBackendKind::from_env("HNSW"),
            Some(SearchBackendKind::Hnsw)
        );
        assert_eq!(SearchBackendKind::from_env("invalid"), None);
    }
}
