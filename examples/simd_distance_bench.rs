//! Benchmark AVX2 SIMD distance computation vs scalar baseline.
//!
//! Run with: cargo run --release --example simd_distance_bench

use rinha_2026::detection::simd::{distance_squared_scalar, distance_squared};
use std::time::Instant;

const VECTOR_DIMENSIONS: usize = 14;
const ITERATIONS: usize = 1_000_000;

fn main() {
    // Generate test vectors
    let mut query = [0i16; VECTOR_DIMENSIONS];
    let mut reference = [0i16; VECTOR_DIMENSIONS];
    
    for i in 0..VECTOR_DIMENSIONS {
        query[i] = ((i * 1234) % 10000) as i16;
        reference[i] = ((i * 5678) % 10000) as i16;
    }

    println!("Benchmarking distance computation over {} iterations", ITERATIONS);
    println!("Query:     {:?}", &query[..4]);
    println!("Reference: {:?}", &reference[..4]);
    println!();

    // Warmup
    for _ in 0..10_000 {
        let _ = distance_squared_scalar(&query, &reference);
    }

    // Scalar baseline
    let start = Instant::now();
    let mut sum = 0i64;
    for _ in 0..ITERATIONS {
        sum = sum.wrapping_add(distance_squared_scalar(&query, &reference) as i64);
    }
    let scalar_elapsed = start.elapsed();
    println!("Scalar:    {:?} ({:.2} ns/call, checksum: {})", 
        scalar_elapsed,
        scalar_elapsed.as_nanos() as f64 / ITERATIONS as f64,
        sum
    );

    // SIMD (AVX2 on x86_64, scalar fallback elsewhere)
    let start = Instant::now();
    let mut sum = 0i64;
    for _ in 0..ITERATIONS {
        sum = sum.wrapping_add(distance_squared(&query, &reference) as i64);
    }
    let simd_elapsed = start.elapsed();
    println!("SIMD:      {:?} ({:.2} ns/call, checksum: {})", 
        simd_elapsed,
        simd_elapsed.as_nanos() as f64 / ITERATIONS as f64,
        sum
    );

    let speedup = scalar_elapsed.as_nanos() as f64 / simd_elapsed.as_nanos() as f64;
    println!();
    println!("Speedup: {:.2}x", speedup);

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    println!("(AVX2 enabled)");
    
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    println!("(AVX2 not available - using scalar fallback)");
}
