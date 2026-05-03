use super::*;

const MAX_KMEANS_SAMPLE: usize = 50_000;
const MAX_KMEANS_ITERATIONS: usize = 25;



#[derive(Debug)]
struct KMeansModel {
    centroids: Vec<[f32; VECTOR_DIMENSIONS]>,
    assignments: Vec<u16>,
}

#[derive(Debug)]
struct WorkerAccumulation {
    changed: usize,
    counts: Vec<u32>,
    sums: Vec<[f32; VECTOR_DIMENSIONS]>,
}

impl FraudEngine {
    pub fn classify_knn(
        &self,
        query: &[f32; VECTOR_DIMENSIONS],
        neighbors: usize,
    ) -> (usize, usize) {
        search_exact_knn(&self.dataset, query, neighbors)
    }
}


pub fn build_ivf_index(references: &[StoredReference]) -> Result<IvfIndex, FraudEngineError> {
    if references.is_empty() {
        return Ok(IvfIndex {
            reference_count: 0,
            centroids_transposed: Vec::new(),
            radii: Vec::new(),
            block_offsets: vec![0],
            labels: Vec::new(),
            quantized_blocks: Vec::new(),
        });
    }

    let model = train_kmeans(references)?;
    let cluster_count = model.centroids.len();
    let quantized_centroids = quantized_centroids(&model.centroids);

    let mut cluster_sizes = vec![0usize; cluster_count];
    for &assignment in &model.assignments {
        cluster_sizes[assignment as usize] += 1;
    }

    let mut vector_offsets = Vec::with_capacity(cluster_count + 1);
    vector_offsets.push(0usize);
    for &size in &cluster_sizes {
        vector_offsets.push(vector_offsets.last().copied().unwrap_or(0) + size);
    }

    let mut grouped_indices = vec![0usize; references.len()];
    let mut write_positions = vector_offsets[..cluster_count].to_vec();
    let mut radii = vec![0.0f32; cluster_count];

    for (reference_index, reference) in references.iter().enumerate() {
        let cluster = model.assignments[reference_index] as usize;
        let position = write_positions[cluster];
        grouped_indices[position] = reference_index;
        write_positions[cluster] += 1;

        let cluster_centroids = quantized_centroids[cluster];

        let mut distance_sq = 0.0f32;
        for (dimension, &value) in reference.vector.iter().enumerate() {
            let delta = math::quantized_as_f32(value) - cluster_centroids[dimension];
            distance_sq += delta * delta;
        }
        radii[cluster] = radii[cluster].max(distance_sq.sqrt());
    }

    let mut block_offsets = Vec::with_capacity(cluster_count + 1);
    block_offsets.push(0u32);
    for &size in &cluster_sizes {
        let next = block_offsets.last().copied().unwrap_or(0)
            + u32::try_from(size.div_ceil(BLOCK_WIDTH)).map_err(|_| {
                FraudEngineError::Load("cluster block count exceeds u32 range".to_owned())
            })?;
        block_offsets.push(next);
    }

    let total_blocks = block_offsets.last().copied().unwrap_or(0) as usize;
    let mut labels = vec![PADDED_LABEL_VALUE; total_blocks * BLOCK_WIDTH];
    let mut quantized_blocks = vec![i16::MAX; total_blocks * VECTOR_DIMENSIONS * BLOCK_WIDTH];

    for cluster in 0..cluster_count {
        let cluster_start = vector_offsets[cluster];
        let cluster_end = vector_offsets[cluster + 1];
        let block_start = block_offsets[cluster] as usize;

        for (local_index, &reference_index) in grouped_indices[cluster_start..cluster_end]
            .iter()
            .enumerate()
        {
            let block_index = block_start + (local_index / BLOCK_WIDTH);
            let lane = local_index % BLOCK_WIDTH;
            let label_offset = block_index * BLOCK_WIDTH + lane;
            labels[label_offset] = references[reference_index].label.to_storage_byte();

            for dimension in 0..VECTOR_DIMENSIONS {
                let block_value_offset =
                    block_index * VECTOR_DIMENSIONS * BLOCK_WIDTH + dimension * BLOCK_WIDTH + lane;
                quantized_blocks[block_value_offset] =
                    math::quantize(references[reference_index].vector[dimension]);
            }
        }
    }

    Ok(IvfIndex {
        reference_count: references.len(),
        centroids_transposed: transpose_centroids(&quantized_centroids),
        radii,
        block_offsets,
        labels,
        quantized_blocks,
    })
}

fn search_exact_knn(
    dataset: &OwnedDataset,
    query: &[f32; VECTOR_DIMENSIONS],
    neighbors: usize,
) -> (usize, usize) {
    let mut result = topk::SortedTopK::new();
    if dataset.len() == 0 || neighbors == 0 || dataset.cluster_count() == 0 {
        return (0, 0);
    }

    let quantized_query: [i16; VECTOR_DIMENSIONS] =
        std::array::from_fn(|dimension| math::quantize(query[dimension]));
    let query_quantized_f32 = quantized_query.map(|value| value as f32);
    let mut cluster_order = (0..dataset.cluster_count())
        .map(|cluster| {
            (
                cluster,
                centroid_distance_squared(dataset, &query_quantized_f32, cluster),
            )
        })
        .collect::<Vec<_>>();
    cluster_order.sort_unstable_by(|left, right| left.1.total_cmp(&right.1));

    for (cluster, centroid_distance_sq) in cluster_order {
        if result.found >= neighbors {
            let lower_bound = centroid_distance_sq.sqrt() - dataset.radius(cluster);
            let bounded = lower_bound.max(0.0);
            if bounded * bounded > result.worst_distance(neighbors) as f32 {
                continue;
            }
        }

        scan_cluster(dataset, &quantized_query, cluster, neighbors, &mut result);
    }

    result.finalize(neighbors)
}

fn scan_cluster(
    dataset: &OwnedDataset,
    query: &[i16; VECTOR_DIMENSIONS],
    cluster: usize,
    neighbors: usize,
    result: &mut topk::SortedTopK,
) {
    let block_offsets = dataset.block_offsets();
    let labels = dataset.labels();
    let quantized_blocks = dataset.quantized_blocks();
    let block_start = block_offsets[cluster] as usize;
    let block_end = block_offsets[cluster + 1] as usize;

    for block_index in block_start..block_end {
        let block_base = block_index * VECTOR_DIMENSIONS * BLOCK_WIDTH;
        let label_base = block_index * BLOCK_WIDTH;

        for lane in 0..BLOCK_WIDTH {
            let label = labels[label_base + lane];
            if label == PADDED_LABEL_VALUE {
                continue;
            }

            // Use SIMD distance computation with early-exit threshold
            // This skips remaining dimensions if partial distance already exceeds worst found
            let mut reference = [0i16; VECTOR_DIMENSIONS];
            for dimension in 0..VECTOR_DIMENSIONS {
                reference[dimension] = quantized_blocks[block_base + dimension * BLOCK_WIDTH + lane];
            }
            
            // Get current worst distance threshold for early-exit
            let threshold = result.worst_distance(neighbors);
            
            // Compute distance with threshold-based early exit
            let distance = simd::distance_squared_with_threshold(query, &reference, threshold);

            // Only insert if distance is below threshold (i32::MAX indicates early exit)
            if distance != i32::MAX {
                result.insert(distance, label, neighbors);
            }
        }
    }
}

fn centroid_distance_squared(
    dataset: &OwnedDataset,
    query: &[f32; VECTOR_DIMENSIONS],
    cluster: usize,
) -> f32 {
    let mut total = 0.0f32;

    for (dimension, value) in query.iter().enumerate() {
        let delta = value - dataset.centroid_component(dimension, cluster);
        total += delta * delta;
    }

    total
}

fn train_kmeans(references: &[StoredReference]) -> Result<KMeansModel, FraudEngineError> {
    let cluster_count = references.len().min(IVF_CLUSTER_COUNT);
    let sample_size = references.len().min(MAX_KMEANS_SAMPLE).max(cluster_count);
    let mut rng = SplitMix64::seeded(0x51f1_2026_d15c_a11e);
    let sample_indices = reservoir_sample_indices(references.len(), sample_size, &mut rng);
    let sample_vectors = sample_indices
        .iter()
        .map(|&index| references[index].vector)
        .collect::<Vec<_>>();

    let mut centroids = initialize_centroids_kmeans_pp(&sample_vectors, cluster_count, &mut rng);
    let mut assignments = vec![u16::MAX; references.len()];
    let convergence_threshold = (references.len() / 10_000).max(8);

    for _ in 0..MAX_KMEANS_ITERATIONS {
        let worker_count = std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1)
            .min(references.len().max(1));
        let chunk_size = references.len().div_ceil(worker_count);
        let mut next_assignments = vec![0u16; references.len()];
        let mut total_changed = 0usize;
        let mut counts = vec![0u32; cluster_count];
        let mut sums = vec![[0.0; VECTOR_DIMENSIONS]; cluster_count];

        std::thread::scope(|scope| {
            let mut handles = Vec::new();

            for (worker_index, next_chunk) in next_assignments.chunks_mut(chunk_size).enumerate() {
                let start = worker_index * chunk_size;
                let end = start + next_chunk.len();
                let reference_chunk = &references[start..end];
                let previous_chunk = &assignments[start..end];
                let centroid_slice = &centroids;

                handles.push(scope.spawn(move || {
                    assign_chunk(reference_chunk, previous_chunk, next_chunk, centroid_slice)
                }));
            }

            for handle in handles {
                let partial = handle.join().expect("k-means worker thread panicked");
                total_changed += partial.changed;

                for (cluster, count) in partial.counts.into_iter().enumerate() {
                    counts[cluster] += count;
                }

                for (cluster, cluster_sums) in partial.sums.into_iter().enumerate() {
                    for (dimension, value) in cluster_sums.into_iter().enumerate() {
                        sums[cluster][dimension] += value;
                    }
                }
            }
        });

        for cluster in 0..cluster_count {
            if counts[cluster] == 0 {
                continue;
            }

            let scale = 1.0 / counts[cluster] as f32;
            for dimension in 0..VECTOR_DIMENSIONS {
                centroids[cluster][dimension] = sums[cluster][dimension] * scale;
            }
        }

        assignments = next_assignments;
        if total_changed <= convergence_threshold {
            break;
        }
    }

    Ok(KMeansModel {
        centroids,
        assignments,
    })
}

fn assign_chunk(
    references: &[StoredReference],
    previous_assignments: &[u16],
    next_assignments: &mut [u16],
    centroids: &[[f32; VECTOR_DIMENSIONS]],
) -> WorkerAccumulation {
    let mut changed = 0usize;
    let mut counts = vec![0u32; centroids.len()];
    let mut sums = vec![[0.0; VECTOR_DIMENSIONS]; centroids.len()];

    for (index, reference) in references.iter().enumerate() {
        let nearest = nearest_centroid(&reference.vector, centroids);
        let nearest_u16 = nearest as u16;
        next_assignments[index] = nearest_u16;

        if previous_assignments[index] != nearest_u16 {
            changed += 1;
        }

        counts[nearest] += 1;
        for (dimension, value) in reference.vector.iter().enumerate() {
            sums[nearest][dimension] += value;
        }
    }

    WorkerAccumulation {
        changed,
        counts,
        sums,
    }
}

fn initialize_centroids_kmeans_pp(
    sample_vectors: &[[f32; VECTOR_DIMENSIONS]],
    centroid_count: usize,
    rng: &mut SplitMix64,
) -> Vec<[f32; VECTOR_DIMENSIONS]> {
    let mut centroids = Vec::with_capacity(centroid_count);
    centroids.push(sample_vectors[rng.gen_bounded_usize(sample_vectors.len())]);
    let mut nearest_distances = vec![f32::INFINITY; sample_vectors.len()];

    while centroids.len() < centroid_count {
        let newest_centroid = centroids.last().expect("centroid exists");
        for (index, vector) in sample_vectors.iter().enumerate() {
            let distance = math::l2_squared(vector, newest_centroid);
            if distance < nearest_distances[index] {
                nearest_distances[index] = distance;
            }
        }

        let total_weight = nearest_distances.iter().copied().sum::<f32>();
        let next_index = if total_weight <= 0.0 {
            rng.gen_bounded_usize(sample_vectors.len())
        } else {
            let mut target = rng.next_f32() * total_weight;
            let mut selected = sample_vectors.len() - 1;
            for (index, &weight) in nearest_distances.iter().enumerate() {
                target -= weight;
                if target <= 0.0 {
                    selected = index;
                    break;
                }
            }
            selected
        };

        centroids.push(sample_vectors[next_index]);
    }

    centroids
}

fn quantized_centroids(centroids: &[[f32; VECTOR_DIMENSIONS]]) -> Vec<[f32; VECTOR_DIMENSIONS]> {
    centroids
        .iter()
        .map(|centroid| centroid.map(math::quantized_as_f32))
        .collect()
}

fn transpose_centroids(centroids: &[[f32; VECTOR_DIMENSIONS]]) -> Vec<f32> {
    let mut transposed = Vec::with_capacity(VECTOR_DIMENSIONS * centroids.len());

    for dimension in 0..VECTOR_DIMENSIONS {
        for centroid in centroids {
            transposed.push(centroid[dimension]);
        }
    }

    transposed
}

fn nearest_centroid(
    vector: &[f32; VECTOR_DIMENSIONS],
    centroids: &[[f32; VECTOR_DIMENSIONS]],
) -> usize {
    let mut best_index = 0usize;
    let mut best_distance = f32::INFINITY;

    for (index, centroid) in centroids.iter().enumerate() {
        let distance = math::l2_squared(vector, centroid);
        if distance < best_distance {
            best_distance = distance;
            best_index = index;
        }
    }

    best_index
}

fn reservoir_sample_indices(len: usize, sample_size: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let mut sample = (0..sample_size).collect::<Vec<_>>();

    for index in sample_size..len {
        let replacement = rng.gen_bounded_usize(index + 1);
        if replacement < sample_size {
            sample[replacement] = index;
        }
    }

    sample
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn seeded(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        let value = (self.next_u64() >> 40) as u32;
        value as f32 / ((1u32 << 24) as f32)
    }

    fn gen_bounded_usize(&mut self, upper_bound: usize) -> usize {
        if upper_bound <= 1 {
            0
        } else {
            (self.next_u64() % upper_bound as u64) as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn test_reference(vector: [f32; VECTOR_DIMENSIONS], is_fraud: bool) -> StoredReference {
        StoredReference {
            vector,
            label: if is_fraud {
                ReferenceLabel::Fraud
            } else {
                ReferenceLabel::Legit
            },
        }
    }

    fn brute_force_knn(
        references: &[StoredReference],
        query: &[f32; VECTOR_DIMENSIONS],
        neighbors: usize,
    ) -> (usize, usize) {
        let quantized_query: [i16; VECTOR_DIMENSIONS] =
            std::array::from_fn(|dimension| math::quantize(query[dimension]));
        let mut result = topk::SortedTopK::new();

        for reference in references {
            let mut distance = 0i32;
            for (dimension, &value) in reference.vector.iter().enumerate() {
                let delta = math::quantize(value) as i32 - quantized_query[dimension] as i32;
                distance += delta * delta;
            }
            result.insert(distance, reference.label.to_storage_byte(), neighbors);
        }

        result.finalize(neighbors)
    }

    fn synthetic_engine(references: Vec<StoredReference>) -> FraudEngine {
        let index = build_ivf_index(&references).expect("ivf index should build");

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
            dataset: OwnedDataset { index },
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
    fn ivf_quantized_exact_search_matches_bruteforce_knn() {
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
        let engine = synthetic_engine(references.clone());
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
            let brute_force = brute_force_knn(&references, &query, K_NEIGHBORS);

            assert_eq!(exact.0, brute_force.0);
            assert_eq!(exact.1, brute_force.1);
        }
    }

    #[test]
    fn ivf_quantized_exact_search_matches_bruteforce_across_large_random_dataset() {
        let references = generate_random_dataset(1_024, 0x5eed_f00d_dead_beef);
        let engine = synthetic_engine(references.clone());
        let mut query_state = 0x1234_5678_9abc_def0;

        for _ in 0..32 {
            let query = random_query(&mut query_state);
            let exact = search_exact_knn(&engine.dataset, &query, K_NEIGHBORS);
            let brute_force = brute_force_knn(&references, &query, K_NEIGHBORS);

            assert_eq!(exact.0, brute_force.0);
            assert_eq!(exact.1, brute_force.1);
        }
    }
}
