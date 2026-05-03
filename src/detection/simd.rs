//! AVX2 SIMD distance computation for i16 quantized vectors.
//!
//! This module provides optimized squared Euclidean distance computation
//! using AVX2 intrinsics for 14-dimensional i16 vectors.
//!
//! Key patterns:
//! - SSE MADD for i16→i32 squared distance accumulation (8 dimensions)
//! - Scalar loop for remaining 6 dimensions
//! - Batched early-exit: check partial distance after 8D to skip remaining 6D
//! - Horizontal sum via hadd (5 cycles vs 10 for store/reload)

use super::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Compute squared Euclidean distance for first 8 dimensions using SSE MADD.
///
/// # Safety
/// Requires AVX2 support. Caller must verify at runtime or compile-time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn distance_squared_first_8_dims_avx2(
    query: &[i16; VECTOR_DIMENSIONS],
    reference: &[i16; VECTOR_DIMENSIONS],
) -> i32 {
    unsafe {
        // Load first 8 i16 values (128 bits) into XMM register
        let query_v = _mm_loadu_si128(query.as_ptr() as *const __m128i);
        let reference_v = _mm_loadu_si128(reference.as_ptr() as *const __m128i);

        // Compute differences: delta = query - reference
        let delta = _mm_sub_epi16(query_v, reference_v);

        // MADD: multiply adjacent i16 pairs and accumulate into i32
        // result[0] = delta[0]*delta[0] + delta[1]*delta[1]
        // result[1] = delta[2]*delta[2] + delta[3]*delta[3]
        // result[2] = delta[4]*delta[4] + delta[5]*delta[5]
        // result[3] = delta[6]*delta[6] + delta[7]*delta[7]
        let acc = _mm_madd_epi16(delta, delta);

        // Horizontal sum of first 8 dimensions
        let sum = _mm_hadd_epi32(acc, acc);
        let sum = _mm_hadd_epi32(sum, sum);
        _mm_cvtsi128_si32(sum)
    }
}

/// Compute squared Euclidean distance between two i16 vectors using AVX2.
///
/// # Safety
/// Requires AVX2 support. Caller must verify at runtime or compile-time.
///
/// # Correctness
/// - Handles 14D vectors with scale 10000
/// - Max accumulation: 14 × (10000)² = 1.4×10⁹ < i32::MAX
/// - Exact match to scalar implementation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn distance_squared_avx2(
    query: &[i16; VECTOR_DIMENSIONS],
    reference: &[i16; VECTOR_DIMENSIONS],
) -> i32 {
    let mut distance = unsafe { distance_squared_first_8_dims_avx2(query, reference) };

    // Process remaining 6 dimensions (indices 8-13) with scalar loop
    for dimension in 8..VECTOR_DIMENSIONS {
        let delta = query[dimension] as i32 - reference[dimension] as i32;
        distance = distance.wrapping_add(delta * delta);
    }

    distance
}

/// Compute squared Euclidean distance with early-exit threshold.
///
/// Returns the distance if it's below the threshold, or i32::MAX if it exceeds.
/// This allows callers to skip expensive full distance computation.
///
/// # Safety
/// Requires AVX2 support. Caller must verify at runtime or compile-time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn distance_squared_with_threshold_avx2(
    query: &[i16; VECTOR_DIMENSIONS],
    reference: &[i16; VECTOR_DIMENSIONS],
    threshold: i32,
) -> i32 {
    // Compute first 8 dimensions
    let partial = unsafe { distance_squared_first_8_dims_avx2(query, reference) };

    // Early exit if partial distance already exceeds threshold
    if partial > threshold {
        return i32::MAX;
    }

    // Compute remaining 6 dimensions
    let mut distance = partial;
    for dimension in 8..VECTOR_DIMENSIONS {
        let delta = query[dimension] as i32 - reference[dimension] as i32;
        distance = distance.wrapping_add(delta * delta);

        // Check threshold after each dimension for additional early-exit opportunities
        if distance > threshold {
            return i32::MAX;
        }
    }

    distance
}

/// Fallback scalar implementation for non-AVX2 targets or verification.
#[inline]
pub fn distance_squared_scalar(
    query: &[i16; VECTOR_DIMENSIONS],
    reference: &[i16; VECTOR_DIMENSIONS],
) -> i32 {
    let mut distance = 0i32;
    for dimension in 0..VECTOR_DIMENSIONS {
        let delta = query[dimension] as i32 - reference[dimension] as i32;
        distance = distance.wrapping_add(delta * delta);
    }
    distance
}

/// Scalar version with early-exit threshold.
#[inline]
pub fn distance_squared_with_threshold_scalar(
    query: &[i16; VECTOR_DIMENSIONS],
    reference: &[i16; VECTOR_DIMENSIONS],
    threshold: i32,
) -> i32 {
    let mut distance = 0i32;
    for dimension in 0..VECTOR_DIMENSIONS {
        let delta = query[dimension] as i32 - reference[dimension] as i32;
        distance = distance.wrapping_add(delta * delta);

        // Early exit if threshold exceeded
        if distance > threshold {
            return i32::MAX;
        }
    }
    distance
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn distance_squared_block_with_threshold_avx2(
    query: &[i16; VECTOR_DIMENSIONS],
    block: &[i16],
    threshold: i32,
) -> [i32; BLOCK_WIDTH] {
    debug_assert_eq!(block.len(), VECTOR_DIMENSIONS * BLOCK_WIDTH);

    unsafe {
        let mut acc = _mm256_setzero_si256();

        for (dimension, &query_value) in query.iter().enumerate() {
            let row_offset = dimension * BLOCK_WIDTH;
            let references_i16 =
                _mm_loadu_si128(block[row_offset..].as_ptr() as *const __m128i);
            let references = _mm256_cvtepi16_epi32(references_i16);
            let query_lane = _mm256_set1_epi32(query_value as i32);
            let delta = _mm256_sub_epi32(query_lane, references);
            let squared = _mm256_mullo_epi32(delta, delta);
            acc = _mm256_add_epi32(acc, squared);
        }

        let mut distances = [0i32; BLOCK_WIDTH];
        _mm256_storeu_si256(distances.as_mut_ptr() as *mut __m256i, acc);

        if threshold != i32::MAX {
            for distance in &mut distances {
                if *distance > threshold {
                    *distance = i32::MAX;
                }
            }
        }

        distances
    }
}

#[inline]
pub fn distance_squared_block_with_threshold_scalar(
    query: &[i16; VECTOR_DIMENSIONS],
    block: &[i16],
    threshold: i32,
) -> [i32; BLOCK_WIDTH] {
    debug_assert_eq!(block.len(), VECTOR_DIMENSIONS * BLOCK_WIDTH);

    let mut distances = [0i32; BLOCK_WIDTH];
    for (lane, distance) in distances.iter_mut().enumerate() {
        let mut total = 0i32;
        for (dimension, &query_value) in query.iter().enumerate() {
            let delta = query_value as i32 - block[dimension * BLOCK_WIDTH + lane] as i32;
            total = total.wrapping_add(delta * delta);
            if total > threshold {
                total = i32::MAX;
                break;
            }
        }
        *distance = total;
    }

    distances
}

/// Select the appropriate distance function based on CPU capabilities.
#[inline]
pub fn distance_squared(
    query: &[i16; VECTOR_DIMENSIONS],
    reference: &[i16; VECTOR_DIMENSIONS],
) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { distance_squared_avx2(query, reference) };
        }
    }

    distance_squared_scalar(query, reference)
}

/// Select the appropriate distance function with threshold based on CPU capabilities.
#[inline]
pub fn distance_squared_with_threshold(
    query: &[i16; VECTOR_DIMENSIONS],
    reference: &[i16; VECTOR_DIMENSIONS],
    threshold: i32,
) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { distance_squared_with_threshold_avx2(query, reference, threshold) };
        }
    }

    distance_squared_with_threshold_scalar(query, reference, threshold)
}

#[inline]
pub fn distance_squared_block_with_threshold(
    query: &[i16; VECTOR_DIMENSIONS],
    block: &[i16],
    threshold: i32,
) -> [i32; BLOCK_WIDTH] {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { distance_squared_block_with_threshold_avx2(query, block, threshold) };
        }
    }

    distance_squared_block_with_threshold_scalar(query, block, threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avx2_matches_scalar() {
        let query: [i16; VECTOR_DIMENSIONS] = [
            100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400,
        ];
        let reference: [i16; VECTOR_DIMENSIONS] = [
            150, 250, 350, 450, 550, 650, 750, 850, 950, 1050, 1150, 1250, 1350, 1450,
        ];

        let scalar_result = distance_squared_scalar(&query, &reference);

        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                let avx2_result = unsafe { distance_squared_avx2(&query, &reference) };
                assert_eq!(
                    scalar_result, avx2_result,
                    "AVX2 and scalar results must match exactly"
                );
            }
        }

        // Verify correctness: (150-100)² + (250-200)² + ... = 50² × 14 = 35000
        let expected = 50i32 * 50 * 14;
        assert_eq!(scalar_result, expected);
    }

    #[test]
    fn distance_zero_for_identical_vectors() {
        let vector: [i16; VECTOR_DIMENSIONS] = [100; VECTOR_DIMENSIONS];
        let result = distance_squared_scalar(&vector, &vector);
        assert_eq!(result, 0);

        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                let avx2_result = unsafe { distance_squared_avx2(&vector, &vector) };
                assert_eq!(avx2_result, 0);
            }
        }
    }

    #[test]
    fn distance_no_overflow_at_scale_10000() {
        // Max distance: 14 × (10000)² = 1.4×10⁹ < i32::MAX (2.147×10⁹)
        // Use realistic values within quantization scale
        let query: [i16; VECTOR_DIMENSIONS] = [5000; VECTOR_DIMENSIONS];
        let reference: [i16; VECTOR_DIMENSIONS] = [-5000; VECTOR_DIMENSIONS];

        let result = distance_squared_scalar(&query, &reference);
        assert!(result > 0, "Distance should be positive");
        // (5000 - (-5000))² × 14 = 10000² × 14 = 1.4×10⁹
        let expected = (10000i32 * 10000) * 14;
        assert_eq!(result, expected);

        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                let avx2_result = unsafe { distance_squared_avx2(&query, &reference) };
                assert_eq!(result, avx2_result);
            }
        }
    }

    #[test]
    fn threshold_early_exit_works() {
        let query: [i16; VECTOR_DIMENSIONS] = [
            100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400,
        ];
        let reference: [i16; VECTOR_DIMENSIONS] = [
            150, 250, 350, 450, 550, 650, 750, 850, 950, 1050, 1150, 1250, 1350, 1450,
        ];

        let full_distance = distance_squared_scalar(&query, &reference);
        assert_eq!(full_distance, 35000);

        // Threshold below full distance should return i32::MAX
        let result_below = distance_squared_with_threshold_scalar(&query, &reference, 30000);
        assert_eq!(result_below, i32::MAX);

        // Threshold above full distance should return full distance
        let result_above = distance_squared_with_threshold_scalar(&query, &reference, 40000);
        assert_eq!(result_above, full_distance);

        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                let avx2_below = unsafe {
                    distance_squared_with_threshold_avx2(&query, &reference, 30000)
                };
                assert_eq!(avx2_below, i32::MAX);

                let avx2_above = unsafe {
                    distance_squared_with_threshold_avx2(&query, &reference, 40000)
                };
                assert_eq!(avx2_above, full_distance);
            }
        }
    }

    #[test]
    fn block_distance_matches_scalar_per_lane() {
        let query: [i16; VECTOR_DIMENSIONS] = [
            100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400,
        ];
        let block: [i16; VECTOR_DIMENSIONS * BLOCK_WIDTH] =
            std::array::from_fn(|index| (index as i16 % 17) * 25 - 200);

        let block_distances = distance_squared_block_with_threshold_scalar(&query, &block, i32::MAX);

        for lane in 0..BLOCK_WIDTH {
            let reference = std::array::from_fn(|dimension| block[dimension * BLOCK_WIDTH + lane]);
            assert_eq!(block_distances[lane], distance_squared_scalar(&query, &reference));
        }

        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                let avx2_distances =
                    unsafe { distance_squared_block_with_threshold_avx2(&query, &block, i32::MAX) };
                assert_eq!(avx2_distances, block_distances);
            }
        }
    }
}
