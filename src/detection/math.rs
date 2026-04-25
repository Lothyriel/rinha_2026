use super::*;

#[inline]
pub fn normalize_amount_vs_avg(amount: f32, customer_average: f32, scaling_ratio: f32) -> f32 {
    if customer_average <= 0.0 {
        return if amount <= 0.0 { 0.0 } else { 1.0 };
    }

    clamp_unit((amount / customer_average) / scaling_ratio)
}

#[inline]
pub fn clamp_ratio(value: f32, max_value: f32) -> f32 {
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
pub fn bool_to_unit(value: bool) -> f32 {
    if value { 1.0 } else { 0.0 }
}

impl Vec16 {
    pub fn from_vector(vector: [f32; VECTOR_DIMENSIONS]) -> Self {
        let mut padded = [0.0; PADDED_VECTOR_DIMENSIONS];
        padded[..VECTOR_DIMENSIONS].copy_from_slice(&vector);
        Self(padded)
    }

    pub fn from_query(vector: &[f32; VECTOR_DIMENSIONS]) -> Self {
        let mut padded = [0.0; PADDED_VECTOR_DIMENSIONS];
        padded[..VECTOR_DIMENSIONS].copy_from_slice(vector);
        Self(padded)
    }
}

impl ExactKnnResult {
    pub fn new() -> Self {
        Self {
            best_distances: [f32::INFINITY; K_NEIGHBORS],
            best_indices: [VP_NONE; K_NEIGHBORS],
            found: 0,
        }
    }

    pub fn insert(&mut self, idx: u32, distance: f32, neighbors: usize) {
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

        debug_assert!(
            self.best_distances
                .windows(2)
                .all(|window| window[0] <= window[1])
        );
    }

    pub fn worst_distance(&self, neighbors: usize) -> f32 {
        if self.found < neighbors {
            f32::INFINITY
        } else {
            self.best_distances[neighbors - 1]
        }
    }

    pub fn fraud_votes(&self, references: &[StoredReference]) -> usize {
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
pub fn l2_squared_scalar(left: &Vec16, right: &Vec16) -> f32 {
    let mut total = 0.0f32;

    for index in 0..PADDED_VECTOR_DIMENSIONS {
        let difference = left.0[index] - right.0[index];
        total += difference * difference;
    }

    total
}
#[target_feature(enable = "avx")]
pub fn l2_squared_avx(left: &Vec16, right: &Vec16) -> f32 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_sub_ps};

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

#[inline]
#[target_feature(enable = "avx,sse,sse3")]
unsafe fn horizontal_sum_m256(
    #[cfg(target_arch = "x86_64")] value: std::arch::x86_64::__m256,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps, _mm256_castps256_ps128, _mm256_extractf128_ps,
    };

    let lower = _mm256_castps256_ps128(value);
    let upper = _mm256_extractf128_ps(value, 1);
    let combined = _mm_add_ps(lower, upper);
    let sum = _mm_hadd_ps(combined, combined);
    let sum = _mm_hadd_ps(sum, sum);
    _mm_cvtss_f32(sum)
}

#[inline]
pub fn normalize_l2(vector: &[f32; VECTOR_DIMENSIONS]) -> [f32; VECTOR_DIMENSIONS] {
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
