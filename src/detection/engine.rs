use std::{path::Path, time::Instant};

use metrics::histogram;

use crate::model::{FraudScoreRequest, FraudScoreResponse};

use super::*;

impl FraudEngine {
    pub fn load(
        resources_dir: &Path,
        configured_search_backend: SearchBackendKind,
        hnsw_config: HnswConfig,
    ) -> Result<Self, FraudEngineError> {
        let normalization = loader::load_json_file(resources_dir.join("normalization.json"))?;
        let mcc_risk = loader::load_json_file(resources_dir.join("mcc_risk.json"))?;
        let references = loader::load_references(resources_dir)?;

        if references.len() < K_NEIGHBORS {
            return Err(FraudEngineError::Load(format!(
                "reference dataset must contain at least {K_NEIGHBORS} vectors, found {}",
                references.len()
            )));
        }

        let search_index = search::build_search_index(&references, configured_search_backend, hnsw_config);

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

        histogram!("score_engine", "step" => "classify")
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
            SearchIndex::Exact(_) => SearchBackendKind::Exact.as_str(),
            SearchIndex::Hnsw(_) => SearchBackendKind::Hnsw.as_str(),
        }
    }
}
