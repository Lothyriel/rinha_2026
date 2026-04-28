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

impl QuantizedVec16 {
    pub fn from_vector(vector: [f32; VECTOR_DIMENSIONS]) -> Self {
        let mut padded = [0i16; PADDED_VECTOR_DIMENSIONS];

        for (index, value) in vector.into_iter().enumerate() {
            padded[index] = quantize_component(value);
        }

        Self(padded)
    }

    pub fn from_query(vector: &[f32; VECTOR_DIMENSIONS]) -> Self {
        let mut padded = [0i16; PADDED_VECTOR_DIMENSIONS];

        for (index, value) in vector.iter().copied().enumerate() {
            padded[index] = quantize_component(value);
        }

        Self(padded)
    }
}

#[inline]
fn quantize_component(value: f32) -> i16 {
    let clamped = value.clamp(-1.0, 1.0);
    let scaled = (clamped * QUANTIZATION_SCALE).round();
    scaled.clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

impl ExactKnnResult {
    pub fn new() -> Self {
        Self {
            best_distances: [u64::MAX; K_NEIGHBORS],
            best_indices: [VP_NONE; K_NEIGHBORS],
            found: 0,
        }
    }

    pub fn insert(&mut self, idx: u32, distance: u64, neighbors: usize) {
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

    pub fn worst_distance(&self, neighbors: usize) -> u64 {
        if self.found < neighbors {
            u64::MAX
        } else {
            self.best_distances[neighbors - 1]
        }
    }

    pub fn fraud_votes(&self, dataset: &DatasetStorage) -> usize {
        self.best_indices[..self.found]
            .iter()
            .filter(|&&idx| {
                idx != VP_NONE
                    && dataset.label(idx as usize) == ReferenceLabel::Fraud
            })
            .count()
    }
}

#[inline]
pub fn l2_squared_scalar(left: &QuantizedVec16, right: &QuantizedVec16) -> u64 {
    let mut total = 0u64;

    for index in 0..PADDED_VECTOR_DIMENSIONS {
        let difference = left.0[index] as i64 - right.0[index] as i64;
        total += (difference * difference) as u64;
    }

    total
}
