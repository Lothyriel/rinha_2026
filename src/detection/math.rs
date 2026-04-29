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

impl QuantizedVector {
    pub fn from_vector(vector: [f32; VECTOR_DIMENSIONS]) -> Self {
        let mut padded = [0; STORED_VECTOR_DIMENSIONS];
        for (index, value) in vector.iter().enumerate() {
            padded[index] = quantize_value(*value);
        }
        Self(padded)
    }

    pub fn from_query(vector: &[f32; VECTOR_DIMENSIONS]) -> Self {
        let mut padded = [0; STORED_VECTOR_DIMENSIONS];
        for (index, value) in vector.iter().enumerate() {
            padded[index] = quantize_value(*value);
        }
        Self(padded)
    }
}

#[inline]
fn quantize_value(value: f32) -> i16 {
    let scaled = value.clamp(-1.0, 1.0) * QUANTIZATION_SCALE;
    scaled.round() as i16
}

impl ExactKnnResult {
    pub fn new() -> Self {
        Self {
            best_distances: [i64::MAX; K_NEIGHBORS],
            best_indices: [u32::MAX; K_NEIGHBORS],
            found: 0,
        }
    }

    pub fn insert(&mut self, idx: u32, distance: i64, neighbors: usize) {
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

    pub fn fraud_votes(&self, dataset: &DatasetStorage) -> usize {
        self.best_indices[..self.found]
            .iter()
            .filter(|&&idx| idx != u32::MAX && dataset.label(idx as usize) == ReferenceLabel::Fraud)
            .count()
    }

    pub fn worst_distance(&self, neighbors: usize) -> i64 {
        if self.found < neighbors {
            i64::MAX
        } else {
            self.best_distances[neighbors - 1]
        }
    }
}

#[cfg(test)]
#[inline]
pub fn l2_squared_scalar(left: &QuantizedVector, right: &QuantizedVector) -> i64 {
    let mut total = 0i64;

    for index in 0..STORED_VECTOR_DIMENSIONS {
        let difference = left.0[index] as i32 - right.0[index] as i32;
        total += (difference * difference) as i64;
    }

    total
}

#[inline]
pub fn l2_squared_first_half_scalar(left: &QuantizedVector, right: &QuantizedVector) -> i64 {
    let mut total = 0i64;

    for index in 0..8 {
        let difference = left.0[index] as i32 - right.0[index] as i32;
        total += (difference * difference) as i64;
    }

    total
}

#[inline]
pub fn l2_squared_second_half_scalar(left: &QuantizedVector, right: &QuantizedVector) -> i64 {
    let mut total = 0i64;

    for index in 8..STORED_VECTOR_DIMENSIONS {
        let difference = left.0[index] as i32 - right.0[index] as i32;
        total += (difference * difference) as i64;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn l2_squared_first_half_x86(left: &QuantizedVector, right: &QuantizedVector) -> i64 {
    use std::arch::x86_64::*;

    let left_half = unsafe { _mm_loadu_si128(left.0.as_ptr().cast::<__m128i>()) };
    let right_half = unsafe { _mm_loadu_si128(right.0.as_ptr().cast::<__m128i>()) };
    let diff = _mm_sub_epi16(left_half, right_half);
    let pair_sums = _mm_madd_epi16(diff, diff);
    let mut lanes = [0i32; 4];
    unsafe {
        _mm_storeu_si128(lanes.as_mut_ptr().cast::<__m128i>(), pair_sums);
    }
    lanes.into_iter().map(i64::from).sum()
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn l2_squared_second_half_x86(left: &QuantizedVector, right: &QuantizedVector) -> i64 {
    use std::arch::x86_64::*;

    let left_half = unsafe { _mm_loadu_si128(left.0.as_ptr().add(8).cast::<__m128i>()) };
    let right_half = unsafe { _mm_loadu_si128(right.0.as_ptr().add(8).cast::<__m128i>()) };
    let diff = _mm_sub_epi16(left_half, right_half);
    let pair_sums = _mm_madd_epi16(diff, diff);
    let mut lanes = [0i32; 4];
    unsafe {
        _mm_storeu_si128(lanes.as_mut_ptr().cast::<__m128i>(), pair_sums);
    }
    lanes.into_iter().map(i64::from).sum()
}
