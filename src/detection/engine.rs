use std::path::Path;

use crate::model::{FraudScoreRequest, FraudScoreResponse};

use super::*;

impl FraudEngine {
    pub fn load(resources_dir: &Path) -> Result<Self, FraudEngineError> {
        Self::load_with_leaf_size(resources_dir, LEAF_SIZE)
    }

    pub fn load_with_leaf_size(
        resources_dir: &Path,
        leaf_size: usize,
    ) -> Result<Self, FraudEngineError> {
        let normalization = loader::load_json_file(resources_dir.join("normalization.json"))?;
        let mcc_risk = loader::load_json_file(resources_dir.join("mcc_risk.json"))?;

        let dataset = if let Some(shared_path) = std::env::var("RINHA_SHARED_MMAP_PATH")
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
        {
            DatasetStorage::Shared(shared::load_or_create_mapped_dataset(
                resources_dir,
                Path::new(&shared_path),
                leaf_size,
            )?)
        } else {
            let references = loader::load_refs(resources_dir)?;
            let index = search::build_index(&references, leaf_size);

            DatasetStorage::Owned(OwnedDataset { references, index })
        };

        Self::try_new(dataset, normalization, mcc_risk, leaf_size)
    }

    pub fn load_example(resources_dir: &Path) -> Result<Self, FraudEngineError> {
        Self::load_example_with_leaf_size(resources_dir, LEAF_SIZE)
    }

    pub fn load_example_with_leaf_size(
        resources_dir: &Path,
        leaf_size: usize,
    ) -> Result<Self, FraudEngineError> {
        let normalization = loader::load_json_file(resources_dir.join("normalization.json"))?;
        let mcc_risk = loader::load_json_file(resources_dir.join("mcc_risk.json"))?;
        let references = loader::load_example_refs(resources_dir)?;
        let index = search::build_index(&references, leaf_size);

        Self::try_new(
            DatasetStorage::Owned(OwnedDataset { references, index }),
            normalization,
            mcc_risk,
            leaf_size,
        )
    }

    fn try_new(
        dataset: DatasetStorage,
        normalization: NormalizationConfig,
        mcc_risk: HashMap<String, f32>,
        leaf_size: usize,
    ) -> Result<Self, FraudEngineError> {
        if leaf_size == 0 {
            return Err(FraudEngineError::Load(
                "exact leaf size must be greater than zero".to_owned(),
            ));
        }

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
