//! Verify AVX2 SIMD distance computation correctness.

use rinha_2026::detection::simd::{distance_squared_scalar, distance_squared};

const VECTOR_DIMENSIONS: usize = 14;

fn main() {
    // Test vectors
    let query: [i16; VECTOR_DIMENSIONS] = [
        100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400,
    ];
    let reference: [i16; VECTOR_DIMENSIONS] = [
        150, 250, 350, 450, 550, 650, 750, 850, 950, 1050, 1150, 1250, 1350, 1450,
    ];

    let scalar = distance_squared_scalar(&query, &reference);
    let simd = distance_squared(&query, &reference);

    println!("Query:     {:?}", &query[..4]);
    println!("Reference: {:?}", &reference[..4]);
    println!();
    println!("Scalar result: {}", scalar);
    println!("SIMD result:   {}", simd);
    println!("Match: {}", scalar == simd);

    // Test with random vectors
    println!("\nRandom test vectors:");
    for seed in 0..5 {
        let mut q = [0i16; VECTOR_DIMENSIONS];
        let mut r = [0i16; VECTOR_DIMENSIONS];
        
        for i in 0..VECTOR_DIMENSIONS {
            q[i] = ((seed * 1234 + i as u32 * 5678) % 10000) as i16;
            r[i] = ((seed * 5678 + i as u32 * 1234) % 10000) as i16;
        }
        
        let s = distance_squared_scalar(&q, &r);
        let v = distance_squared(&q, &r);
        
        println!("  Seed {}: scalar={}, simd={}, match={}", seed, s, v, s == v);
    }
}
