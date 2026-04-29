use std::path::Path;

use crate::model::{FraudScoreRequest, FraudScoreResponse};

use super::*;

impl FraudEngine {
    pub fn load(resources_dir: &Path) -> Result<Self, FraudEngineError> {
        let normalization = loader::load_json_file(resources_dir.join("normalization.json"))?;
        let mcc_risk = loader::load_json_file(resources_dir.join("mcc_risk.json"))?;
        let dataset = DatasetStorage::Embedded(shared::load_embedded_dataset()?);

        Self::try_new(dataset, normalization, mcc_risk)
    }

    pub fn load_example(resources_dir: &Path) -> Result<Self, FraudEngineError> {
        let normalization = loader::load_json_file(resources_dir.join("normalization.json"))?;
        let mcc_risk = loader::load_json_file(resources_dir.join("mcc_risk.json"))?;
        let references = loader::load_example_refs(resources_dir)?;

        Self::try_new(
            DatasetStorage::Owned(owned_dataset_from_references(references)),
            normalization,
            mcc_risk,
        )
    }

    fn try_new(
        dataset: DatasetStorage,
        normalization: NormalizationConfig,
        mcc_risk: HashMap<String, f32>,
    ) -> Result<Self, FraudEngineError> {
        if dataset.len() < K_NEIGHBORS {
            return Err(FraudEngineError::Load(format!(
                "reference dataset must contain at least {K_NEIGHBORS} vectors, found {}",
                dataset.len()
            )));
        }

        Ok(Self {
            normalization,
            mcc_risk,
            dataset,
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
        self.dataset.len()
    }
}

fn owned_dataset_from_references(references: Vec<StoredReference>) -> OwnedDataset {
    let vectors = references
        .iter()
        .map(|reference| reference.quantized_vector)
        .collect();
    let labels = references
        .iter()
        .map(|reference| reference.label.to_storage_byte())
        .collect();

    OwnedDataset { vectors, labels }
}
