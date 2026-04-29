use super::*;
use std::{
    fs::{self, OpenOptions},
    mem::{align_of, size_of},
    path::Path,
    slice, thread,
};

pub fn reduce_references_with_kmeans(
    references: &[StoredReference],
    options: DatasetBuildOptions,
) -> Vec<StoredReference> {
    if !options.clustering_enabled_for(references.len()) {
        tracing::info!(
            source_reference_count = references.len(),
            configured_kmeans_k = options.configured_kmeans_k(),
            effective_reference_count = references.len(),
            "kmeans reduction disabled; keeping full reference dataset"
        );
        return references.to_vec();
    }

    let (fraud, legit): (Vec<_>, Vec<_>) = references
        .iter()
        .copied()
        .partition(|reference| reference.label == ReferenceLabel::Fraud);

    let total = references.len();
    let total_k = options.effective_k(total);
    if total_k >= total {
        tracing::info!(
            source_reference_count = total,
            configured_kmeans_k = options.configured_kmeans_k(),
            effective_reference_count = total,
            "kmeans effective K covers full dataset; keeping all references"
        );
        return references.to_vec();
    }

    let fraud_k = if fraud.is_empty() {
        0
    } else {
        let ratio = fraud.len() as f32 / total as f32;
        (total_k as f32 * ratio).round() as usize
    }
    .max(usize::from(!fraud.is_empty()))
    .min(fraud.len());

    let legit_k = total_k
        .saturating_sub(fraud_k)
        .max(usize::from(!legit.is_empty()))
        .min(legit.len());

    tracing::info!(
        source_reference_count = total,
        source_fraud_count = fraud.len(),
        source_legit_count = legit.len(),
        configured_kmeans_k = options.configured_kmeans_k(),
        effective_reference_count = total_k,
        fraud_k,
        legit_k,
        seed = options.kmeans_seed(),
        "starting kmeans representative reduction"
    );

    let fraud_seed = options.seed() ^ 0xF00D;
    let legit_seed = options.seed() ^ 0x1E617;

    let fraud_task = thread::spawn(move || cluster_class(&fraud, fraud_k, fraud_seed, "fraud"));
    let legit_task = thread::spawn(move || cluster_class(&legit, legit_k, legit_seed, "legit"));

    let fraud_reduced = fraud_task
        .join()
        .expect("fraud clustering thread should not panic");
    let legit_reduced = legit_task
        .join()
        .expect("legit clustering thread should not panic");

    let mut reduced = Vec::with_capacity(fraud_reduced.len() + legit_reduced.len());
    reduced.extend(fraud_reduced);
    reduced.extend(legit_reduced);

    tracing::info!(
        source_reference_count = total,
        reduced_reference_count = reduced.len(),
        "finished kmeans representative reduction"
    );
    reduced
}

const DATASET_MAGIC: [u8; 8] = *b"R26MMAP\0";
const DATASET_VERSION: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DatasetHeader {
    magic: [u8; 8],
    version: u32,
    logical_dimensions: u32,
    stored_dimensions: u32,
    quantization_scale: f32,
    clustering_enabled: u32,
    reference_count: u64,
    source_reference_count: u64,
    configured_kmeans_k: u64,
    kmeans_max_iterations: u64,
    kmeans_seed: u64,
    vectors_offset: u64,
    labels_offset: u64,
    file_len: u64,
}

pub fn build_dataset_file(
    resources_dir: &Path,
    output_path: &Path,
    options: DatasetBuildOptions,
) -> Result<(), FraudEngineError> {
    tracing::info!(
        output_path = %output_path.display(),
        resources_dir = %resources_dir.display(),
        configured_kmeans_k = options.configured_kmeans_k(),
        seed = options.kmeans_seed(),
        "building embedded dataset file"
    );

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            FraudEngineError::Load(format!(
                "failed to create dataset directory {}: {error}",
                parent.display()
            ))
        })?;
    }

    let source_references = loader::load_refs(resources_dir)?;
    let references = reduce_references_with_kmeans(&source_references, options);

    if references.len() < K_NEIGHBORS {
        return Err(FraudEngineError::Load(format!(
            "reference dataset must contain at least {K_NEIGHBORS} vectors, found {}",
            references.len()
        )));
    }

    write_dataset_file(output_path, &references, source_references.len(), options)
}

fn cluster_class(
    references: &[StoredReference],
    k: usize,
    seed: u64,
    class_name: &str,
) -> Vec<StoredReference> {
    if references.is_empty() || k == 0 {
        tracing::info!(
            class = class_name,
            requested_k = k,
            "skipping empty class during kmeans"
        );
        return Vec::new();
    }

    if k >= references.len() {
        tracing::info!(
            class = class_name,
            requested_k = k,
            source_reference_count = references.len(),
            "class representative count covers all references; keeping class as-is"
        );
        return references.to_vec();
    }

    tracing::info!(
        class = class_name,
        source_reference_count = references.len(),
        representative_count = k,
        seed,
        "starting class kmeans"
    );

    let mut rng = SimpleRng::new(seed);
    let vectors = references
        .iter()
        .map(|reference| dequantize_vector(&reference.quantized_vector))
        .collect::<Vec<_>>();

    let mut centroid_indices = pick_initial_centroids(&vectors, k, &mut rng);
    let mut centroids = centroid_indices
        .iter()
        .map(|&index| vectors[index])
        .collect::<Vec<_>>();
    let mut assignments = vec![0usize; vectors.len()];

    let mut iteration = 0usize;
    loop {
        iteration += 1;
        let mut changed = 0usize;

        for (index, vector) in vectors.iter().enumerate() {
            let nearest = nearest_centroid(vector, &centroids);
            if assignments[index] != nearest {
                assignments[index] = nearest;
                changed += 1;
            }
        }

        if changed == 0 {
            tracing::info!(
                class = class_name,
                iteration,
                changed_assignments = changed,
                "class kmeans converged"
            );
            break;
        }

        tracing::info!(
            class = class_name,
            iteration,
            changed_assignments = changed,
            "class kmeans iteration complete"
        );

        let mut sums = vec![[0.0f32; VECTOR_DIMENSIONS]; k];
        let mut counts = vec![0usize; k];

        for (index, vector) in vectors.iter().enumerate() {
            let cluster = assignments[index];
            counts[cluster] += 1;
            for dimension in 0..VECTOR_DIMENSIONS {
                sums[cluster][dimension] += vector[dimension];
            }
        }

        for cluster in 0..k {
            if counts[cluster] == 0 {
                let replacement = rng.next_usize() % vectors.len();
                centroids[cluster] = vectors[replacement];
                centroid_indices[cluster] = replacement;
                tracing::warn!(
                    class = class_name,
                    cluster,
                    replacement_index = replacement,
                    "empty class cluster reinitialized from random vector"
                );
                continue;
            }

            let count = counts[cluster] as f32;
            for dimension in 0..VECTOR_DIMENSIONS {
                centroids[cluster][dimension] = sums[cluster][dimension] / count;
            }
        }
    }

    let medoid_indices = medoid_indices(&vectors, &assignments, &centroids);
    tracing::info!(
        class = class_name,
        medoid_count = medoid_indices.len(),
        "finished class medoid selection"
    );

    medoid_indices
        .into_iter()
        .map(|index| references[index])
        .collect()
}

fn write_dataset_file(
    path: &Path,
    references: &[StoredReference],
    source_reference_count: usize,
    options: DatasetBuildOptions,
) -> Result<(), FraudEngineError> {
    let vectors_offset = align_up(
        size_of::<DatasetHeader>() as u64,
        align_of::<QuantizedVector>() as u64,
    );
    let labels_offset = align_up(
        vectors_offset + byte_len::<QuantizedVector>(references.len())?,
        align_of::<u8>() as u64,
    );
    let file_len = labels_offset + byte_len::<u8>(references.len())?;

    let header = DatasetHeader {
        magic: DATASET_MAGIC,
        version: DATASET_VERSION,
        logical_dimensions: VECTOR_DIMENSIONS as u32,
        stored_dimensions: STORED_VECTOR_DIMENSIONS as u32,
        quantization_scale: QUANTIZATION_SCALE,
        clustering_enabled: u32::from(options.clustering_enabled(source_reference_count)),
        reference_count: references.len() as u64,
        source_reference_count: source_reference_count as u64,
        configured_kmeans_k: options.configured_kmeans_k() as u64,
        kmeans_max_iterations: 0,
        kmeans_seed: options.kmeans_seed(),
        vectors_offset,
        labels_offset,
        file_len,
    };

    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .read(true)
        .open(path)
        .map_err(|error| {
            FraudEngineError::Load(format!(
                "failed to create dataset file {}: {error}",
                path.display()
            ))
        })?;

    file.set_len(file_len).map_err(|error| {
        FraudEngineError::Load(format!(
            "failed to size dataset file {}: {error}",
            path.display()
        ))
    })?;

    let bytes = unsafe {
        std::slice::from_raw_parts(
            (&header as *const DatasetHeader).cast::<u8>(),
            size_of::<DatasetHeader>(),
        )
    };
    std::os::unix::fs::FileExt::write_all_at(&file, bytes, 0).map_err(|error| {
        FraudEngineError::Load(format!(
            "failed to write dataset header {}: {error}",
            path.display()
        ))
    })?;

    let vectors = references
        .iter()
        .map(|reference| reference.quantized_vector)
        .collect::<Vec<_>>();
    let vector_bytes = unsafe {
        slice::from_raw_parts(
            vectors.as_ptr().cast::<u8>(),
            std::mem::size_of_val(vectors.as_slice()),
        )
    };
    std::os::unix::fs::FileExt::write_all_at(&file, vector_bytes, vectors_offset).map_err(
        |error| {
            FraudEngineError::Load(format!(
                "failed to write dataset vectors {}: {error}",
                path.display()
            ))
        },
    )?;

    let labels = references
        .iter()
        .map(|reference| reference.label.to_storage_byte())
        .collect::<Vec<_>>();
    std::os::unix::fs::FileExt::write_all_at(&file, &labels, labels_offset).map_err(|error| {
        FraudEngineError::Load(format!(
            "failed to write dataset labels {}: {error}",
            path.display()
        ))
    })?;

    file.sync_all().map_err(|error| {
        FraudEngineError::Load(format!(
            "failed to sync dataset file {}: {error}",
            path.display()
        ))
    })
}

fn byte_len<T>(len: usize) -> Result<u64, FraudEngineError> {
    (len as u64)
        .checked_mul(size_of::<T>() as u64)
        .ok_or_else(|| FraudEngineError::Load("dataset size overflow".to_owned()))
}

fn align_up(offset: u64, alignment: u64) -> u64 {
    let mask = alignment - 1;
    (offset + mask) & !mask
}

fn pick_initial_centroids(
    vectors: &[[f32; VECTOR_DIMENSIONS]],
    k: usize,
    rng: &mut SimpleRng,
) -> Vec<usize> {
    let mut indices = (0..vectors.len()).collect::<Vec<_>>();
    shuffle(&mut indices, rng);
    indices.truncate(k);
    indices
}

fn medoid_indices(
    vectors: &[[f32; VECTOR_DIMENSIONS]],
    assignments: &[usize],
    centroids: &[[f32; VECTOR_DIMENSIONS]],
) -> Vec<usize> {
    let mut medoids = Vec::with_capacity(centroids.len());

    for (cluster, centroid) in centroids.iter().enumerate() {
        let mut best_index = None;
        let mut best_distance = f32::MAX;

        for (index, vector) in vectors.iter().enumerate() {
            if assignments[index] != cluster {
                continue;
            }

            let distance = l2_squared_float(vector, centroid);
            if distance < best_distance {
                best_distance = distance;
                best_index = Some(index);
            }
        }

        if let Some(index) = best_index {
            medoids.push(index);
        }
    }

    medoids
}

fn nearest_centroid(
    vector: &[f32; VECTOR_DIMENSIONS],
    centroids: &[[f32; VECTOR_DIMENSIONS]],
) -> usize {
    let mut nearest = 0usize;
    let mut nearest_distance = f32::MAX;

    for (index, centroid) in centroids.iter().enumerate() {
        let distance = l2_squared_float(vector, centroid);
        if distance < nearest_distance {
            nearest_distance = distance;
            nearest = index;
        }
    }

    nearest
}

fn l2_squared_float(left: &[f32; VECTOR_DIMENSIONS], right: &[f32; VECTOR_DIMENSIONS]) -> f32 {
    let mut total = 0.0f32;
    for index in 0..VECTOR_DIMENSIONS {
        let difference = left[index] - right[index];
        total += difference * difference;
    }
    total
}

fn dequantize_vector(vector: &QuantizedVector) -> [f32; VECTOR_DIMENSIONS] {
    std::array::from_fn(|index| vector.0[index] as f32 / QUANTIZATION_SCALE)
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.state
    }

    fn next_usize(&mut self) -> usize {
        (self.next() >> 32) as usize
    }
}

fn shuffle<T>(slice: &mut [T], rng: &mut SimpleRng) {
    for index in (1..slice.len()).rev() {
        let other = rng.next_usize() % (index + 1);
        slice.swap(index, other);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fraud_reference(base: f32) -> StoredReference {
        StoredReference {
            quantized_vector: QuantizedVector::from_vector([base; VECTOR_DIMENSIONS]),
            label: ReferenceLabel::Fraud,
        }
    }

    fn legit_reference(base: f32) -> StoredReference {
        StoredReference {
            quantized_vector: QuantizedVector::from_vector([base; VECTOR_DIMENSIONS]),
            label: ReferenceLabel::Legit,
        }
    }

    #[test]
    fn kmeans_reduction_preserves_both_classes() {
        let mut references = Vec::new();
        for value in 0..32 {
            references.push(fraud_reference(0.05 + value as f32 * 0.001));
            references.push(legit_reference(0.85 - value as f32 * 0.001));
        }

        let reduced =
            reduce_references_with_kmeans(&references, DatasetBuildOptions::fixed_clustered(8, 42));

        assert!(reduced.len() <= 8);
        assert!(
            reduced
                .iter()
                .any(|reference| reference.label == ReferenceLabel::Fraud)
        );
        assert!(
            reduced
                .iter()
                .any(|reference| reference.label == ReferenceLabel::Legit)
        );
    }
}
