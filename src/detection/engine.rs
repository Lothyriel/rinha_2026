use std::path::Path;

use crate::model::{FraudScoreRequest, FraudScoreResponse};

use super::*;

impl FraudEngine {
    pub fn load(resources_dir: &Path) -> Result<Self, FraudEngineError> {
        let normalization = loader::load_json_file(resources_dir.join("normalization.json"))?;
        let mcc_risk = loader::load_json_file(resources_dir.join("mcc_risk.json"))?;
        let references = loader::load_references(resources_dir)?;

        if references.len() < K_NEIGHBORS {
            return Err(FraudEngineError::Load(format!(
                "reference dataset must contain at least {K_NEIGHBORS} vectors, found {}",
                references.len()
            )));
        }

        let index = search::build_index(&references);

        Ok(Self {
            normalization,
            mcc_risk,
            references,
            index,
        })
    }

    pub fn score(
        &self,
        request: &FraudScoreRequest,
    ) -> Result<FraudScoreResponse, FraudEngineError> {
        let vector = self.vectorize(request)?;

        let (neighbor_count, fraud_votes) = self.classify_knn(&vector, K_NEIGHBORS);

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
}
