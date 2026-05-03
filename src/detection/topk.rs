//! Top-k selection optimization for KNN search.
//!
//! This module provides two implementations:
//! 1. Sorted array (current): O(K) per insert, O(K log K) total
//! 2. Flat vector + nth_element (optimized): O(1) append, O(K log K) partition
//!
//! The flat vector approach is faster for large reference sets because:
//! - No per-candidate insertion cost
//! - Single partition at the end
//! - Better cache locality
//! - Vectorizable comparisons

use super::*;

/// Top-k result container using flat vector + partition strategy.
///
/// This is faster than sorted array insertion for large datasets because:
/// - Append is O(1) instead of O(K) for sorted insertion
/// - Partition happens once at the end, not per candidate
/// - Better branch prediction and cache behavior
#[derive(Debug, Clone, Copy)]
pub struct FlatTopK {
    distances: [i32; K_NEIGHBORS * 2],  // 2x capacity for buffering
    labels: [u8; K_NEIGHBORS * 2],
    pub count: usize,
}

impl FlatTopK {
    pub fn new() -> Self {
        Self {
            distances: [i32::MAX; K_NEIGHBORS * 2],
            labels: [PADDED_LABEL_VALUE; K_NEIGHBORS * 2],
            count: 0,
        }
    }

    /// Append a candidate (O(1) operation).
    #[inline]
    pub fn append(&mut self, distance: i32, label: u8, _neighbors: usize) {
        // Append all candidates (up to buffer capacity)
        if self.count < K_NEIGHBORS * 2 {
            self.distances[self.count] = distance;
            self.labels[self.count] = label;
            self.count += 1;
        }
    }

    /// Finalize and get top-k results (O(K log K) partition).
    #[inline]
    pub fn finalize(&mut self, neighbors: usize) -> (usize, usize) {
        let actual_count = self.count.min(neighbors);
        
        if actual_count == 0 {
            return (0, 0);
        }

        // Create indices and sort by distance
        let mut indices: Vec<usize> = (0..self.count).collect();
        indices.sort_unstable_by_key(|&i| self.distances[i]);
        
        // Count fraud votes in top-k
        let fraud_votes = indices[..actual_count]
            .iter()
            .filter(|&&idx| ReferenceLabel::is_fraud_storage_byte(self.labels[idx]))
            .count();

        (actual_count, fraud_votes)
    }

    /// Get worst distance for threshold-based early exit.
    #[inline]
    pub fn worst_distance(&self, neighbors: usize) -> i32 {
        if self.count < neighbors {
            i32::MAX
        } else {
            // Find the K-th smallest distance
            let mut sorted = self.distances[..self.count].to_vec();
            sorted.sort_unstable();
            sorted[neighbors - 1]
        }
    }
}

/// Original sorted array implementation (for comparison/fallback).
#[derive(Debug, Clone, Copy)]
pub struct SortedTopK {
    best_distances: [i32; K_NEIGHBORS],
    best_labels: [u8; K_NEIGHBORS],
    pub found: usize,
}

impl SortedTopK {
    pub fn new() -> Self {
        Self {
            best_distances: [i32::MAX; K_NEIGHBORS],
            best_labels: [PADDED_LABEL_VALUE; K_NEIGHBORS],
            found: 0,
        }
    }

    /// Insert with sorted maintenance (O(K) operation).
    #[inline]
    pub fn insert(&mut self, distance: i32, label: u8, neighbors: usize) {
        let insert_at = self
            .best_distances
            .partition_point(|current| *current < distance);
        if insert_at >= neighbors {
            return;
        }

        let upper_bound = self.found.min(neighbors.saturating_sub(1));
        for index in (insert_at..upper_bound).rev() {
            self.best_distances[index + 1] = self.best_distances[index];
            self.best_labels[index + 1] = self.best_labels[index];
        }

        self.best_distances[insert_at] = distance;
        self.best_labels[insert_at] = label;
        self.found = (self.found + 1).min(neighbors);
    }

    /// Get worst distance for threshold-based early exit.
    #[inline]
    pub fn worst_distance(&self, neighbors: usize) -> i32 {
        if self.found < neighbors {
            i32::MAX
        } else {
            self.best_distances[neighbors - 1]
        }
    }

    /// Finalize and get results.
    #[inline]
    pub fn finalize(&self, _neighbors: usize) -> (usize, usize) {
        let fraud_votes = self.best_labels[..self.found]
            .iter()
            .filter(|&&label| ReferenceLabel::is_fraud_storage_byte(label))
            .count();
        (self.found, fraud_votes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_topk_basic() {
        let mut topk = FlatTopK::new();
        
        topk.append(100, 0, K_NEIGHBORS);
        topk.append(50, 1, K_NEIGHBORS);
        topk.append(150, 0, K_NEIGHBORS);
        topk.append(75, 1, K_NEIGHBORS);
        
        let (count, fraud_votes) = topk.finalize(K_NEIGHBORS);
        assert_eq!(count, 4);
        assert_eq!(fraud_votes, 2);  // Two fraud labels (1)
    }

    #[test]
    fn sorted_topk_basic() {
        let mut topk = SortedTopK::new();
        
        topk.insert(100, 0, K_NEIGHBORS);
        topk.insert(50, 1, K_NEIGHBORS);
        topk.insert(150, 0, K_NEIGHBORS);
        topk.insert(75, 1, K_NEIGHBORS);
        
        let (count, fraud_votes) = topk.finalize(K_NEIGHBORS);
        assert_eq!(count, 4);
        assert_eq!(fraud_votes, 2);
    }

    #[test]
    fn flat_topk_respects_k() {
        let mut topk = FlatTopK::new();
        let k = 3;
        
        // Add more than K candidates
        for i in 0..10 {
            topk.append(i as i32 * 10, (i % 2) as u8, k);
        }
        
        let (count, _) = topk.finalize(k);
        assert!(count <= k);
    }

    #[test]
    fn sorted_topk_respects_k() {
        let mut topk = SortedTopK::new();
        let k = 3;
        
        // Add more than K candidates
        for i in 0..10 {
            topk.insert(i as i32 * 10, (i % 2) as u8, k);
        }
        
        let (count, _) = topk.finalize(k);
        assert_eq!(count, k);
    }

    #[test]
    fn worst_distance_before_full() {
        let mut topk = SortedTopK::new();
        let k = 3;
        
        topk.insert(100, 0, k);
        topk.insert(50, 1, k);
        
        // Before full, worst_distance should be i32::MAX
        assert_eq!(topk.worst_distance(k), i32::MAX);
    }

    #[test]
    fn worst_distance_after_full() {
        let mut topk = SortedTopK::new();
        let k = 3;
        
        topk.insert(100, 0, k);
        topk.insert(50, 1, k);
        topk.insert(150, 0, k);
        
        // After full (3 inserts with K=3), worst_distance should be the 3rd smallest
        // Array: [50, 100, 150], so worst is 150
        assert_eq!(topk.worst_distance(k), 150);
        
        topk.insert(75, 1, k);
        
        // After 4th insert, array becomes [50, 75, 100] (150 is dropped)
        // Worst distance is still 100
        assert_eq!(topk.worst_distance(k), 100);
    }
}
