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
        let result = search_exact_knn(&self.references, &self.index, query, neighbors);
        (result.found, result.fraud_votes(&self.references))
    }
}

pub fn build_index(references: &[StoredReference]) -> ExactSearchIndex {
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
            dist: math::l2_squared_scalar(pivot, &references[idx as usize].padded_vector),
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

            total_distance += math::l2_squared_scalar(
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

pub fn search_exact_knn(
    references: &[StoredReference],
    index: &ExactSearchIndex,
    query: &[f32; VECTOR_DIMENSIONS],
    neighbors: usize,
) -> ExactKnnResult {
    let query = Vec16::from_query(query);

    search_exact_knn_with_distance(references, index, &query, neighbors, |left, right| unsafe {
        math::l2_squared_avx(left, right)
    })
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

        let can_visit_far = if far == VP_NONE {
            false
        } else if result.found < neighbors {
            true
        } else {
            let worst_distance = result.worst_distance(neighbors);
            (pivot_distance.sqrt() - node.radius.sqrt()).abs() <= worst_distance.sqrt() + EPSILON
        };

        if can_visit_far {
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn test_reference(vector: [f32; VECTOR_DIMENSIONS], is_fraud: bool) -> StoredReference {
        StoredReference {
            padded_vector: Vec16::from_vector(vector),
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
    ) -> ExactKnnResult {
        let query = Vec16::from_query(query);
        let mut result = ExactKnnResult::new();

        for (index, reference) in references.iter().enumerate() {
            result.insert(
                index as u32,
                math::l2_squared_scalar(&query, &reference.padded_vector),
                neighbors,
            );
        }

        result
    }

    fn synthetic_engine(references: Vec<StoredReference>) -> FraudEngine {
        let index = build_index(&references);

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
            index,
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
    fn vp_tree_exact_search_matches_bruteforce_knn() {
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
            let exact = search_exact_knn(&engine.references, &engine.index, &query, K_NEIGHBORS);
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

        let index = &engine.index;
        let exact = search_exact_knn(&engine.references, index, &query, K_NEIGHBORS);

        assert_eq!(exact.found, K_NEIGHBORS);
        assert_eq!(index.nodes.len(), 1);
        assert_eq!(index.nodes[0].len, 48);
        assert!(
            exact.best_distances[..exact.found]
                .iter()
                .all(|distance| *distance == 0.0)
        );
    }

    #[test]
    fn vp_tree_exact_search_matches_bruteforce_across_large_random_dataset() {
        let engine = synthetic_engine(generate_random_dataset(4_096, 0x5eed_f00d_dead_beef));
        let mut query_state = 0x1234_5678_9abc_def0;

        for _ in 0..256 {
            let query = random_query(&mut query_state);
            let exact = search_exact_knn(&engine.references, &engine.index, &query, K_NEIGHBORS);
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
}
