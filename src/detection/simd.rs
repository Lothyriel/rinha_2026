//! AVX2 SIMD distance computation for i16 quantized vectors.
//!
//! This module provides optimized squared Euclidean distance computation
//! using AVX2 intrinsics for 14-dimensional i16 vectors.
//!
//! Key patterns:
//! - VPMADDWD for i16→i32 squared distance accumulation (8 dimensions)
//! - Scalar loop for remaining 6 dimensions
//! - Horizontal sum via extract + hadd (5 cycles vs 10 for store/reload)

use super::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use std::arch::x86_64::*;

/// Compute squared Euclidean distance between two i16 vectors using AVX2.
///
/// # Safety
/// Requires AVX2 support. Caller must verify at runtime or compile-time.
///
/// # Correctness
/// - Handles 14D vectors with scale 10000
/// - Max accumulation: 14 × (10000)² = 1.4×10⁹ < i32::MAX
/// - Exact match to scalar implementation
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
pub fn distance_squared_avx2(
    query: &[i16; VECTOR_DIMENSIONS],
    reference: &[i16; VECTOR_DIMENSIONS],
) -> i32 {
    unsafe {
        // Process first 8 dimensions with VPMADDWD
        // Load only 8 i16 values (128 bits) into XMM register
        let query_v = _mm_loadu_si128(query.as_ptr() as *const __m128i);
        let reference_v = _mm_loadu_si128(reference.as_ptr() as *const __m128i);

        // Compute differences: delta = query - reference
        let delta = _mm_sub_epi16(query_v, reference_v);

        // VPMADDWD: multiply adjacent i16 pairs and accumulate into i32
        // result[0] = delta[0]*delta[0] + delta[1]*delta[1]
        // result[1] = delta[2]*delta[2] + delta[3]*delta[3]
        // result[2] = delta[4]*delta[4] + delta[5]*delta[5]
        // result[3] = delta[6]*delta[6] + delta[7]*delta[7]
        let acc = _mm_madd_epi16(delta, delta);

        // Horizontal sum of first 8 dimensions
        let sum = _mm_hadd_epi32(acc, acc);
        let sum = _mm_hadd_epi32(sum, sum);
        let mut distance = _mm_cvtsi128_si32(sum);

        // Process remaining 6 dimensions (indices 8-13) with scalar loop
        for dimension in 8..VECTOR_DIMENSIONS {
            let delta = query[dimension] as i32 - reference[dimension] as i32;
            distance = distance.wrapping_add(delta * delta);
        }

        distance
    }
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

/// Select the appropriate distance function based on CPU capabilities.
#[inline]
pub fn distance_squared(
    query: &[i16; VECTOR_DIMENSIONS],
    reference: &[i16; VECTOR_DIMENSIONS],
) -> i32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        distance_squared_avx2(query, reference)
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        distance_squared_scalar(query, reference)
    }
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

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            let avx2_result = distance_squared_avx2(&query, &reference);
            assert_eq!(
                scalar_result, avx2_result,
                "AVX2 and scalar results must match exactly"
            );
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

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            let avx2_result = distance_squared_avx2(&vector, &vector);
            assert_eq!(avx2_result, 0);
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

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            let avx2_result = distance_squared_avx2(&query, &reference);
            assert_eq!(result, avx2_result);
        }
    }
}
