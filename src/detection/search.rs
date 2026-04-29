use super::*;

impl FraudEngine {
    pub fn classify_knn(
        &self,
        query: &[f32; VECTOR_DIMENSIONS],
        neighbors: usize,
    ) -> (usize, usize) {
        self.classify_exact(query, neighbors)
    }

    fn classify_exact(&self, query: &[f32; VECTOR_DIMENSIONS], neighbors: usize) -> (usize, usize) {
        let result = search_exact_knn(&self.dataset, query, neighbors);
        (result.found, result.fraud_votes(&self.dataset))
    }
}

pub fn search_exact_knn(
    dataset: &DatasetStorage,
    query: &[f32; VECTOR_DIMENSIONS],
    neighbors: usize,
) -> ExactKnnResult {
    let query = QuantizedVector::from_query(query);

    if neighbors == 0 || dataset.len() == 0 {
        return ExactKnnResult::new();
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { search_exact_knn_avx2(dataset, &query, neighbors) };
        }
    }

    search_exact_knn_scalar(dataset, &query, neighbors)
}

fn search_exact_knn_scalar(
    dataset: &DatasetStorage,
    query: &QuantizedVector,
    neighbors: usize,
) -> ExactKnnResult {
    let mut result = ExactKnnResult::new();

    for index in 0..dataset.len() {
        let candidate = dataset.vector(index);
        let first_half = math::l2_squared_first_half_scalar(query, candidate);

        if first_half > result.worst_distance(neighbors) {
            continue;
        }

        let distance = first_half + math::l2_squared_second_half_scalar(query, candidate);
        result.insert(index as u32, distance, neighbors);
    }

    result
}

#[cfg(target_arch = "x86_64")]
unsafe fn search_exact_knn_avx2(
    dataset: &DatasetStorage,
    query: &QuantizedVector,
    neighbors: usize,
) -> ExactKnnResult {
    use std::arch::x86_64::_MM_HINT_T0;

    let mut result = ExactKnnResult::new();

    for index in 0..dataset.len() {
        if index + 8 < dataset.len() {
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    dataset.vector(index + 8).0.as_ptr().cast::<i8>(),
                    _MM_HINT_T0,
                );
            }
        }

        let candidate = dataset.vector(index);
        let first_half = unsafe { math::l2_squared_first_half_x86(query, candidate) };

        if first_half > result.worst_distance(neighbors) {
            continue;
        }

        let distance = first_half + unsafe { math::l2_squared_second_half_x86(query, candidate) };
        result.insert(index as u32, distance, neighbors);
    }

    result
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn test_reference(vector: [f32; VECTOR_DIMENSIONS], is_fraud: bool) -> StoredReference {
        StoredReference {
            quantized_vector: QuantizedVector::from_vector(vector),
            label: if is_fraud {
                ReferenceLabel::Fraud
            } else {
                ReferenceLabel::Legit
            },
        }
    }

    fn brute_force_knn(
        dataset: &DatasetStorage,
        query: &[f32; VECTOR_DIMENSIONS],
        neighbors: usize,
    ) -> ExactKnnResult {
        let query = QuantizedVector::from_query(query);
        let mut result = ExactKnnResult::new();

        for index in 0..dataset.len() {
            result.insert(
                index as u32,
                math::l2_squared_scalar(&query, dataset.vector(index)),
                neighbors,
            );
        }

        result
    }

    fn synthetic_engine(references: Vec<StoredReference>) -> FraudEngine {
        let vectors = references
            .iter()
            .map(|reference| reference.quantized_vector)
            .collect();
        let labels = references
            .iter()
            .map(|reference| reference.label.to_storage_byte())
            .collect();

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
            dataset: DatasetStorage::Owned(OwnedDataset { vectors, labels }),
        }
    }

    fn next_random_unit(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);

        ((*state >> 32) as u32) as f32 / u32::MAX as f32
    }

    fn generate_random_dataset(len: usize, seed: u64) -> Vec<StoredReference> {
        let mut state = seed;

        (0..len)
            .map(|index| {
                let mut vector = [0.0; VECTOR_DIMENSIONS];

                for value in &mut vector {
                    *value = next_random_unit(&mut state);
                }

                test_reference(vector, index % 3 == 0)
            })
            .collect()
    }

    fn random_query(state: &mut u64) -> [f32; VECTOR_DIMENSIONS] {
        let mut query = [0.0; VECTOR_DIMENSIONS];

        for value in &mut query {
            *value = next_random_unit(state);
        }

        query
    }

    #[test]
    fn exact_search_matches_bruteforce_knn() {
        let references = (0..96u32)
            .map(|value| {
                let vector = std::array::from_fn(|dimension| {
                    ((value * (dimension as u32 + 3)) % 29) as f32 / 31.0
                        + value as f32 / 100.0
                        + dimension as f32 / 1_000.0
                });

                test_reference(vector, value % 3 == 0)
            })
            .collect::<Vec<_>>();
        let engine = synthetic_engine(references);
        let queries = [
            [
                0.11, 0.07, 0.43, 0.29, 0.31, 0.17, 0.23, 0.19, 0.41, 0.13, 0.37, 0.47, 0.53, 0.59,
            ],
            [
                0.62, 0.18, 0.24, 0.76, 0.08, 0.44, 0.52, 0.34, 0.68, 0.12, 0.56, 0.26, 0.72, 0.16,
            ],
            [
                0.91, 0.83, 0.75, 0.67, 0.59, 0.51, 0.43, 0.35, 0.27, 0.19, 0.11, 0.03, 0.95, 0.87,
            ],
        ];

        for query in queries {
            let exact = search_exact_knn(&engine.dataset, &query, K_NEIGHBORS);
            let brute_force = brute_force_knn(&engine.dataset, &query, K_NEIGHBORS);

            assert_eq!(exact.found, brute_force.found);
            assert_eq!(exact.best_indices, brute_force.best_indices);
            assert_eq!(exact.best_distances, brute_force.best_distances);
        }
    }

    #[test]
    fn exact_search_handles_identical_vectors() {
        let references = (0..48u32)
            .map(|value| test_reference([0.25; VECTOR_DIMENSIONS], value % 2 == 0))
            .collect::<Vec<_>>();
        let engine = synthetic_engine(references);
        let query = [0.25; VECTOR_DIMENSIONS];

        let exact = search_exact_knn(&engine.dataset, &query, K_NEIGHBORS);

        assert_eq!(exact.found, K_NEIGHBORS);
        assert!(
            exact.best_distances[..exact.found]
                .iter()
                .all(|distance| *distance == 0)
        );
    }

    #[test]
    fn exact_search_matches_bruteforce_across_large_random_dataset() {
        let engine = synthetic_engine(generate_random_dataset(4_096, 0x5eed_f00d_dead_beef));
        let mut query_state = 0x1234_5678_9abc_def0;

        for _ in 0..256 {
            let query = random_query(&mut query_state);
            let exact = search_exact_knn(&engine.dataset, &query, K_NEIGHBORS);
            let brute_force = brute_force_knn(&engine.dataset, &query, K_NEIGHBORS);

            assert_eq!(exact.found, brute_force.found);
            assert_eq!(exact.best_indices, brute_force.best_indices);
            assert_eq!(exact.best_distances, brute_force.best_distances);
        }
    }
}
