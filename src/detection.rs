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
const VECTOR_DIMENSIONS: usize = 14;
const PADDED_VECTOR_DIMENSIONS: usize = 16;
const LEAF_SIZE: usize = 32;
const VP_NONE: u32 = u32::MAX;
const EXACT_STACK_CAPACITY: usize = 256;
const PIVOT_SAMPLE_SIZE: usize = 64;

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
            SearchIndex::Exact(_) => "exact",
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
            SearchIndex::Exact(index) => self.classify_exact(query, neighbors, index),
            SearchIndex::Hnsw(index) => self.classify_hnsw(query, neighbors, index),
        };

        histogram!(
            "score_engine",
            "step" => "classify"
        )
        .record(start.elapsed().as_micros() as f64);

        result
    }

    fn classify_exact(
        &self,
        query: &[f32; VECTOR_DIMENSIONS],
        neighbors: usize,
        index: &ExactSearchIndex,
    ) -> (usize, usize) {
        let result = search_exact_knn(&self.references, index, query, neighbors);
        (result.found, result.fraud_votes(&self.references))
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
        SearchBackendKind::Exact => SearchIndex::Exact(build_exact_index(references)),
        SearchBackendKind::Hnsw => match build_hnsw_index(references, hnsw_config) {
            Ok(index) => SearchIndex::Hnsw(index),
            Err(error) => {
                tracing::error!(
                    ?error,
                    "failed to build HNSW index; falling back to exact search"
                );
                SearchIndex::Exact(build_exact_index(references))
            }
        },
    }
}

fn build_hnsw_index(
    references: &[StoredReference],
    hnsw_config: HnswConfig,
) -> Result<HnswSearchIndex, FraudEngineError> {
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

fn build_exact_index(references: &[StoredReference]) -> ExactSearchIndex {
    let mut nodes = Vec::new();
    let mut indices = Vec::with_capacity(references.len());
    let mut point_indices = (0..references.len() as u32).collect::<Vec<_>>();

    if !point_indices.is_empty() {
        build_vp_node(references, &mut point_indices, &mut nodes, &mut indices, 1);
    }

    ExactSearchIndex { nodes, indices }
}

fn build_vp_node(
    references: &[StoredReference],
    point_indices: &mut [u32],
    nodes: &mut Vec<VpNode>,
    leaf_indices: &mut Vec<u32>,
    depth: usize,
) -> u32 {
    let node_idx = nodes.len() as u32;
    nodes.push(VpNode {
        pivot_idx: VP_NONE,
        radius: 0.0,
        left: VP_NONE,
        right: VP_NONE,
        start: 0,
        len: 0,
    });

    if point_indices.len() <= LEAF_SIZE || depth >= EXACT_STACK_CAPACITY {
        finalize_leaf_node(nodes, node_idx, point_indices, leaf_indices);
        return node_idx;
    }

    let pivot_position = choose_pivot_position(references, point_indices);
    point_indices.swap(pivot_position, point_indices.len() - 1);

    let pivot_idx = point_indices[point_indices.len() - 1];
    let pivot = &references[pivot_idx as usize].padded_vector;
    let candidate_len = point_indices.len() - 1;
    let candidates = &mut point_indices[..candidate_len];

    if candidates.is_empty() {
        finalize_leaf_node(nodes, node_idx, point_indices, leaf_indices);
        return node_idx;
    }

    let mut distances = candidates
        .iter()
        .map(|&idx| PointDistance {
            idx,
            dist: l2_squared_scalar(pivot, &references[idx as usize].padded_vector),
        })
        .collect::<Vec<_>>();

    let median_position = distances.len() / 2;
    distances.select_nth_unstable_by(median_position, |left, right| {
        left.dist.total_cmp(&right.dist)
    });

    let radius = distances[median_position].dist;
    let mut left_count = 0usize;

    for point in &distances {
        if point.dist <= radius {
            candidates[left_count] = point.idx;
            left_count += 1;
        }
    }

    if left_count == 0 || left_count == distances.len() {
        finalize_leaf_node(nodes, node_idx, point_indices, leaf_indices);
        return node_idx;
    }

    let mut write_pos = left_count;

    for point in &distances {
        if point.dist > radius {
            candidates[write_pos] = point.idx;
            write_pos += 1;
        }
    }

    let (left_slice, right_slice) = candidates.split_at_mut(left_count);
    let left = build_vp_node(references, left_slice, nodes, leaf_indices, depth + 1);
    let right = build_vp_node(references, right_slice, nodes, leaf_indices, depth + 1);

    nodes[node_idx as usize] = VpNode {
        pivot_idx,
        radius,
        left,
        right,
        start: 0,
        len: 0,
    };

    node_idx
}

fn finalize_leaf_node(
    nodes: &mut [VpNode],
    node_idx: u32,
    point_indices: &[u32],
    leaf_indices: &mut Vec<u32>,
) {
    let start = leaf_indices.len() as u32;
    leaf_indices.extend_from_slice(point_indices);
    nodes[node_idx as usize] = VpNode {
        pivot_idx: VP_NONE,
        radius: 0.0,
        left: VP_NONE,
        right: VP_NONE,
        start,
        len: point_indices.len() as u32,
    };
}

fn choose_pivot_position(references: &[StoredReference], point_indices: &[u32]) -> usize {
    let sample_len = point_indices.len().min(PIVOT_SAMPLE_SIZE);

    if sample_len <= 1 {
        return 0;
    }

    let step = point_indices.len().div_ceil(sample_len);
    let mut sampled_positions = [0usize; PIVOT_SAMPLE_SIZE];

    for (index, position) in sampled_positions.iter_mut().take(sample_len).enumerate() {
        *position = (index * step).min(point_indices.len() - 1);
    }

    let mut best_position = sampled_positions[0];
    let mut best_mean_distance = f32::NEG_INFINITY;

    for &candidate_position in sampled_positions.iter().take(sample_len) {
        let candidate = &references[point_indices[candidate_position] as usize].padded_vector;
        let mut total_distance = 0.0f32;

        for &comparison_position in sampled_positions.iter().take(sample_len) {
            if comparison_position == candidate_position {
                continue;
            }

            total_distance += l2_squared_scalar(
                candidate,
                &references[point_indices[comparison_position] as usize].padded_vector,
            );
        }

        let mean_distance = total_distance / (sample_len.saturating_sub(1) as f32);

        if mean_distance > best_mean_distance {
            best_mean_distance = mean_distance;
            best_position = candidate_position;
        }
    }

    best_position
}

fn search_exact_knn(
    references: &[StoredReference],
    index: &ExactSearchIndex,
    query: &[f32; VECTOR_DIMENSIONS],
    neighbors: usize,
) -> ExactKnnResult {
    let query = Vec16::from_query(query);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
    {
        // SAFETY: guarded by runtime feature detection.
        unsafe {
            return search_exact_knn_with_distance(references, index, &query, neighbors, |left, right| {
                l2_squared_avx2(left, right)
            });
        }
    }

    search_exact_knn_with_distance(references, index, &query, neighbors, l2_squared_scalar)
}

fn search_exact_knn_with_distance(
    references: &[StoredReference],
    index: &ExactSearchIndex,
    query: &Vec16,
    neighbors: usize,
    distance_fn: impl Fn(&Vec16, &Vec16) -> f32,
) -> ExactKnnResult {
    let mut result = ExactKnnResult::new();

    if index.nodes.is_empty() || neighbors == 0 {
        return result;
    }

    let mut stack = [VP_NONE; EXACT_STACK_CAPACITY];
    let mut stack_len = 1usize;
    stack[0] = 0;

    while stack_len > 0 {
        stack_len -= 1;
        let node = &index.nodes[stack[stack_len] as usize];

        if node.len > 0 {
            let leaf_start = node.start as usize;
            let leaf_end = leaf_start + node.len as usize;

            for &reference_idx in &index.indices[leaf_start..leaf_end] {
                let distance =
                    distance_fn(query, &references[reference_idx as usize].padded_vector);
                result.insert(reference_idx, distance, neighbors);
            }

            continue;
        }

        let pivot_idx = node.pivot_idx as usize;
        let pivot_distance = distance_fn(query, &references[pivot_idx].padded_vector);
        result.insert(node.pivot_idx, pivot_distance, neighbors);

        let (near, far) = if pivot_distance <= node.radius {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        if far != VP_NONE && (pivot_distance - node.radius).abs() <= result.worst_distance(neighbors) {
            push_stack(&mut stack, &mut stack_len, far);
        }

        if near != VP_NONE {
            push_stack(&mut stack, &mut stack_len, near);
        }
    }

    result
}

fn push_stack(stack: &mut [u32; EXACT_STACK_CAPACITY], stack_len: &mut usize, value: u32) {
    assert!(
        *stack_len < EXACT_STACK_CAPACITY,
        "vp-tree traversal stack exceeded capacity"
    );
    stack[*stack_len] = value;
    *stack_len += 1;
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
            padded_vector: Vec16::from_vector(value.vector),
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

impl Vec16 {
    fn from_vector(vector: [f32; VECTOR_DIMENSIONS]) -> Self {
        let mut padded = [0.0; PADDED_VECTOR_DIMENSIONS];
        padded[..VECTOR_DIMENSIONS].copy_from_slice(&vector);
        Self(padded)
    }

    fn from_query(vector: &[f32; VECTOR_DIMENSIONS]) -> Self {
        let mut padded = [0.0; PADDED_VECTOR_DIMENSIONS];
        padded[..VECTOR_DIMENSIONS].copy_from_slice(vector);
        Self(padded)
    }
}

impl ExactKnnResult {
    fn new() -> Self {
        Self {
            best_distances: [f32::INFINITY; K_NEIGHBORS],
            best_indices: [VP_NONE; K_NEIGHBORS],
            found: 0,
        }
    }

    fn insert(&mut self, idx: u32, distance: f32, neighbors: usize) {
        let insert_at = self
            .best_distances
            .partition_point(|current| *current < distance);

        if insert_at >= neighbors {
            return;
        }

        let upper_bound = self.found.min(neighbors.saturating_sub(1));

        for index in (insert_at..upper_bound).rev() {
            self.best_distances[index + 1] = self.best_distances[index];
            self.best_indices[index + 1] = self.best_indices[index];
        }

        self.best_distances[insert_at] = distance;
        self.best_indices[insert_at] = idx;
        self.found = (self.found + 1).min(neighbors);
    }

    fn worst_distance(&self, neighbors: usize) -> f32 {
        if self.found < neighbors {
            f32::INFINITY
        } else {
            self.best_distances[neighbors - 1]
        }
    }

    fn fraud_votes(&self, references: &[StoredReference]) -> usize {
        self.best_indices[..self.found]
            .iter()
            .filter(|&&idx| {
                idx != VP_NONE
                    && references
                        .get(idx as usize)
                        .is_some_and(|reference| reference.is_fraud)
            })
            .count()
    }
}

#[inline]
fn l2_squared_scalar(left: &Vec16, right: &Vec16) -> f32 {
    let mut total = 0.0f32;

    for index in 0..PADDED_VECTOR_DIMENSIONS {
        let difference = left.0[index] - right.0[index];
        total += difference * difference;
    }

    total
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,avx2,fma,sse,sse3")]
unsafe fn l2_squared_avx2(left: &Vec16, right: &Vec16) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_sub_ps,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_sub_ps,
    };

    let mut accumulated = _mm256_setzero_ps();

    unsafe {
        for offset in (0..PADDED_VECTOR_DIMENSIONS).step_by(8) {
            let lhs = _mm256_loadu_ps(left.0.as_ptr().add(offset));
            let rhs = _mm256_loadu_ps(right.0.as_ptr().add(offset));
            let difference = _mm256_sub_ps(lhs, rhs);
            accumulated = _mm256_fmadd_ps(difference, difference, accumulated);
        }

        horizontal_sum_m256(accumulated)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx,sse,sse3")]
unsafe fn horizontal_sum_m256(
    #[cfg(target_arch = "x86")] value: std::arch::x86::__m256,
    #[cfg(target_arch = "x86_64")] value: std::arch::x86_64::__m256,
) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        _mm256_castps256_ps128, _mm256_extractf128_ps, _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        _mm256_castps256_ps128, _mm256_extractf128_ps, _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps,
    };

    let lower = _mm256_castps256_ps128(value);
    let upper = _mm256_extractf128_ps(value, 1);
    let combined = _mm_add_ps(lower, upper);
    let sum = _mm_hadd_ps(combined, combined);
    let sum = _mm_hadd_ps(sum, sum);
    _mm_cvtss_f32(sum)
}

#[inline]
fn normalize_l2(vector: &[f32; VECTOR_DIMENSIONS]) -> [f32; VECTOR_DIMENSIONS] {
    let mut squared_norm = 0.0;

    for value in vector {
        squared_norm += value * value;
    }

    if squared_norm <= f32::EPSILON {
        return *vector;
    }

    let norm = squared_norm.sqrt();
    let mut normalized = [0.0; VECTOR_DIMENSIONS];

    for index in 0..VECTOR_DIMENSIONS {
        normalized[index] = vector[index] / norm;
    }

    normalized
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Customer, LastTransaction, Merchant, Terminal, Transaction};

    fn test_reference(vector: [f32; VECTOR_DIMENSIONS], is_fraud: bool) -> StoredReference {
        StoredReference {
            vector,
            padded_vector: Vec16::from_vector(vector),
            is_fraud,
        }
    }

    fn brute_force_knn(
        references: &[StoredReference],
        query: &[f32; VECTOR_DIMENSIONS],
        neighbors: usize,
    ) -> ExactKnnResult {
        let query = Vec16::from_query(query);
        let mut result = ExactKnnResult::new();

        for (index, reference) in references.iter().enumerate() {
            result.insert(
                index as u32,
                l2_squared_scalar(&query, &reference.padded_vector),
                neighbors,
            );
        }

        result
    }

    fn synthetic_engine(references: Vec<StoredReference>) -> FraudEngine {
        let search_index = build_search_index(
            &references,
            SearchBackendKind::Exact,
            HnswConfig::default(),
        );

        FraudEngine {
            normalization: NormalizationConfig {
                max_amount: 1.0,
                max_installments: 1.0,
                amount_vs_avg_ratio: 1.0,
                max_minutes: 1.0,
                max_km: 1.0,
                max_tx_count_24h: 1.0,
                max_merchant_avg_amount: 1.0,
            },
            mcc_risk: HashMap::new(),
            references,
            search_index,
        }
    }

    fn exact_index(engine: &FraudEngine) -> &ExactSearchIndex {
        match &engine.search_index {
            SearchIndex::Exact(index) => index,
            SearchIndex::Hnsw(_) => panic!("expected exact index"),
        }
    }

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

    #[test]
    fn vp_tree_exact_search_matches_bruteforce_knn() {
        let references = (0..96u32)
            .map(|value| {
                let mut vector = [0.0; VECTOR_DIMENSIONS];

                for dimension in 0..VECTOR_DIMENSIONS {
                    vector[dimension] = ((value * (dimension as u32 + 3)) % 29) as f32 / 31.0
                        + value as f32 / 100.0
                        + dimension as f32 / 1_000.0;
                }

                test_reference(vector, value % 3 == 0)
            })
            .collect::<Vec<_>>();
        let engine = synthetic_engine(references);
        let queries = [
            [0.11, 0.07, 0.43, 0.29, 0.31, 0.17, 0.23, 0.19, 0.41, 0.13, 0.37, 0.47, 0.53, 0.59],
            [0.62, 0.18, 0.24, 0.76, 0.08, 0.44, 0.52, 0.34, 0.68, 0.12, 0.56, 0.26, 0.72, 0.16],
            [0.91, 0.83, 0.75, 0.67, 0.59, 0.51, 0.43, 0.35, 0.27, 0.19, 0.11, 0.03, 0.95, 0.87],
        ];

        for query in queries {
            let exact = search_exact_knn(&engine.references, exact_index(&engine), &query, K_NEIGHBORS);
            let brute_force = brute_force_knn(&engine.references, &query, K_NEIGHBORS);

            assert_eq!(exact.found, brute_force.found);
            assert_eq!(exact.best_indices, brute_force.best_indices);

            for (left, right) in exact
                .best_distances
                .iter()
                .zip(brute_force.best_distances.iter())
            {
                assert!((left - right).abs() < 0.0001);
            }
        }
    }

    #[test]
    fn vp_tree_handles_identical_vectors_without_degenerate_recursion() {
        let references = (0..48u32)
            .map(|value| test_reference([0.25; VECTOR_DIMENSIONS], value % 2 == 0))
            .collect::<Vec<_>>();
        let engine = synthetic_engine(references);
        let query = [0.25; VECTOR_DIMENSIONS];

        let index = exact_index(&engine);
        let exact = search_exact_knn(&engine.references, index, &query, K_NEIGHBORS);

        assert_eq!(exact.found, K_NEIGHBORS);
        assert_eq!(index.nodes.len(), 1);
        assert_eq!(index.nodes[0].len, 48);
        assert!(exact.best_distances[..exact.found]
            .iter()
            .all(|distance| *distance == 0.0));
    }
}
