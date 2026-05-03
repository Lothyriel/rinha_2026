//! Debug VPMADDWD behavior

use std::arch::x86_64::*;

fn main() {
    unsafe {
        // Create test vectors
        let query: [i16; 14] = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400];
        let reference: [i16; 14] = [150, 250, 350, 450, 550, 650, 750, 850, 950, 1050, 1150, 1250, 1350, 1450];

        // Load into YMM (256-bit = 16 i16 values, but we only have 14)
        let query_v = _mm256_loadu_si256(query.as_ptr() as *const __m256i);
        let reference_v = _mm256_loadu_si256(reference.as_ptr() as *const __m256i);

        println!("Query loaded into YMM");
        println!("Reference loaded into YMM");

        // Compute differences
        let delta = _mm256_sub_epi16(query_v, reference_v);
        
        // Extract and print delta values
        let delta_array = std::mem::transmute::<__m256i, [i16; 16]>(delta);
        println!("Delta values: {:?}", &delta_array[..14]);
        
        // Apply VPMADDWD
        let acc = _mm256_madd_epi16(delta, delta);
        
        // Extract result
        let acc_array = std::mem::transmute::<__m256i, [i32; 8]>(acc);
        println!("VPMADDWD result: {:?}", acc_array);
        
        // Manual calculation for first 4 pairs
        let expected_0 = (delta_array[0] as i32) * (delta_array[0] as i32) + (delta_array[1] as i32) * (delta_array[1] as i32);
        let expected_1 = (delta_array[2] as i32) * (delta_array[2] as i32) + (delta_array[3] as i32) * (delta_array[3] as i32);
        let expected_2 = (delta_array[4] as i32) * (delta_array[4] as i32) + (delta_array[5] as i32) * (delta_array[5] as i32);
        let expected_3 = (delta_array[6] as i32) * (delta_array[6] as i32) + (delta_array[7] as i32) * (delta_array[7] as i32);
        
        println!("Expected [0]: {}, got: {}", expected_0, acc_array[0]);
        println!("Expected [1]: {}, got: {}", expected_1, acc_array[1]);
        println!("Expected [2]: {}, got: {}", expected_2, acc_array[2]);
        println!("Expected [3]: {}, got: {}", expected_3, acc_array[3]);
    }
}
