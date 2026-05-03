use std::{
    fs::{self, File},
    io::{self, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    thread,
    time::Instant,
};

use flate2::read::GzDecoder;
use serde::Deserialize;

const VECTOR_DIMENSIONS: usize = 14;
const CLUSTER_COUNT: usize = 1_024 * 4;
const BLOCK_WIDTH: usize = 8;
const MAX_KMEANS_SAMPLE: usize = 50_000;
const MAX_KMEANS_ITERATIONS: usize = 25;
const QUANTIZATION_SCALE: f32 = 10_000.0;
const PADDED_QUANTIZED_VALUE: i16 = i16::MAX;
const PADDED_LABEL_VALUE: u8 = u8::MAX;
const DEFAULT_RESOURCES_DIR: &str = "./spec/resources";
const DEFAULT_OUTPUT_PATH: &str = "./index.bin";

type Vector = [f32; VECTOR_DIMENSIONS];

#[derive(Debug, Deserialize)]
struct RawReferenceEntry {
    vector: Vector,
    label: ReferenceLabel,
}

#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum ReferenceLabel {
    Fraud,
    Legit,
}

#[derive(Debug)]
struct Dataset {
    vectors: Vec<Vector>,
    labels: Vec<u8>,
}

#[derive(Debug)]
struct KMeansModel {
    centroids: Vec<Vector>,
    assignments: Vec<u16>,
}

#[derive(Debug)]
struct PackedIndex {
    centroids_transposed: Vec<f32>,
    radii: Vec<f32>,
    block_offsets: Vec<u32>,
    labels: Vec<u8>,
    quantized_blocks: Vec<i16>,
}

#[derive(Debug)]
struct WorkerAccumulation {
    changed: usize,
    counts: Vec<u32>,
    sums: Vec<Vector>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let started_at = Instant::now();
    let resources_dir = PathBuf::from(
        std::env::var("RINHA_RESOURCES_DIR").unwrap_or_else(|_| DEFAULT_RESOURCES_DIR.to_owned()),
    );
    let output_path = PathBuf::from(
        std::env::var("RINHA_INDEX_OUTPUT").unwrap_or_else(|_| DEFAULT_OUTPUT_PATH.to_owned()),
    );

    eprintln!("loading dataset from {}", resources_dir.display());
    let dataset = load_dataset(&resources_dir)?;
    eprintln!("loaded {} vectors", dataset.vectors.len());

    validate_dataset_size(&dataset)?;

    eprintln!(
        "training k-means with k={} (sample <= {})",
        CLUSTER_COUNT, MAX_KMEANS_SAMPLE
    );
    let model = train_kmeans(&dataset.vectors)?;

    eprintln!("packing IVF blocks");
    let packed_index = build_ivf_index(&dataset, &model);

    eprintln!("writing index to {}", output_path.display());
    write_index_file(&output_path, dataset.vectors.len() as u64, &packed_index)?;

    eprintln!(
        "done in {:.2?}: {} clusters, {} blocks",
        started_at.elapsed(),
        CLUSTER_COUNT,
        packed_index.block_offsets.last().copied().unwrap_or(0)
    );

    Ok(())
}

fn validate_dataset_size(dataset: &Dataset) -> io::Result<()> {
    if dataset.vectors.len() != dataset.labels.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "vector/label length mismatch: {} vectors vs {} labels",
                dataset.vectors.len(),
                dataset.labels.len()
            ),
        ));
    }

    if dataset.vectors.len() < CLUSTER_COUNT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "dataset must contain at least {} vectors for k={}, found {}",
                CLUSTER_COUNT,
                CLUSTER_COUNT,
                dataset.vectors.len()
            ),
        ));
    }

    Ok(())
}

fn load_dataset(resources_dir: &Path) -> io::Result<Dataset> {
    let path = resources_dir.join("references.json.gz");
    let file = File::open(&path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("failed to open {}: {error}", path.display()),
        )
    })?;
    let reader = BufReader::new(GzDecoder::new(file));

    let raw_entries: Vec<RawReferenceEntry> = serde_json::from_reader(reader).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to parse {}: {error}", path.display()),
        )
    })?;

    let mut vectors = Vec::with_capacity(raw_entries.len());
    let mut labels = Vec::with_capacity(raw_entries.len());

    for entry in raw_entries {
        vectors.push(entry.vector);
        labels.push(match entry.label {
            ReferenceLabel::Fraud => 0,
            ReferenceLabel::Legit => 1,
        });
    }

    Ok(Dataset { vectors, labels })
}

fn train_kmeans(vectors: &[Vector]) -> io::Result<KMeansModel> {
    let sample_size = vectors.len().min(MAX_KMEANS_SAMPLE);
    if sample_size < CLUSTER_COUNT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "k-means sample must contain at least {CLUSTER_COUNT} points, found {sample_size}"
            ),
        ));
    }

    let mut rng = SplitMix64::seeded(0x51f1_2026_d15c_a11e);
    let sample_indices = reservoir_sample_indices(vectors.len(), sample_size, &mut rng);
    let sample_vectors = sample_indices
        .iter()
        .map(|&index| vectors[index])
        .collect::<Vec<_>>();

    let mut centroids = initialize_centroids_kmeans_pp(&sample_vectors, CLUSTER_COUNT, &mut rng);
    let mut assignments = vec![u16::MAX; vectors.len()];
    let convergence_threshold = (vectors.len() / 10_000).max(128);

    for iteration in 0..MAX_KMEANS_ITERATIONS {
        let iteration_started_at = Instant::now();
        let worker_count = thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1)
            .min(vectors.len().max(1));
        let chunk_size = vectors.len().div_ceil(worker_count);
        let mut next_assignments = vec![0u16; vectors.len()];
        let mut total_changed = 0usize;
        let mut counts = vec![0u32; CLUSTER_COUNT];
        let mut sums = vec![[0.0; VECTOR_DIMENSIONS]; CLUSTER_COUNT];

        thread::scope(|scope| {
            let mut handles = Vec::new();

            for (worker_index, next_chunk) in next_assignments.chunks_mut(chunk_size).enumerate() {
                let start = worker_index * chunk_size;
                let end = start + next_chunk.len();
                let vector_chunk = &vectors[start..end];
                let previous_chunk = &assignments[start..end];
                let centroid_slice = &centroids;

                handles.push(scope.spawn(move || {
                    assign_chunk(vector_chunk, previous_chunk, next_chunk, centroid_slice)
                }));
            }

            for handle in handles {
                let partial = handle.join().expect("k-means worker thread panicked");
                total_changed += partial.changed;

                for (cluster, count) in partial.counts.into_iter().enumerate() {
                    counts[cluster] += count;
                }

                for (cluster, cluster_sums) in partial.sums.into_iter().enumerate() {
                    for (dimension, value) in cluster_sums.into_iter().enumerate() {
                        sums[cluster][dimension] += value;
                    }
                }
            }
        });

        for cluster in 0..CLUSTER_COUNT {
            if counts[cluster] == 0 {
                continue;
            }

            let scale = 1.0 / counts[cluster] as f32;
            for dimension in 0..VECTOR_DIMENSIONS {
                centroids[cluster][dimension] = sums[cluster][dimension] * scale;
            }
        }

        assignments = next_assignments;

        eprintln!(
            "iteration {:02}: changed {} assignments in {:.2?}",
            iteration + 1,
            total_changed,
            iteration_started_at.elapsed()
        );

        if iteration > 0 && total_changed <= convergence_threshold {
            eprintln!(
                "early stop after iteration {:02} (changed <= {})",
                iteration + 1,
                convergence_threshold
            );
            break;
        }
    }

    Ok(KMeansModel {
        centroids,
        assignments,
    })
}

fn assign_chunk(
    vectors: &[Vector],
    previous_assignments: &[u16],
    next_assignments: &mut [u16],
    centroids: &[Vector],
) -> WorkerAccumulation {
    let mut changed = 0usize;
    let mut counts = vec![0u32; centroids.len()];
    let mut sums = vec![[0.0; VECTOR_DIMENSIONS]; centroids.len()];

    for (index, vector) in vectors.iter().enumerate() {
        let nearest = nearest_centroid(vector, centroids);
        let nearest_u16 = nearest as u16;
        next_assignments[index] = nearest_u16;

        if previous_assignments[index] != nearest_u16 {
            changed += 1;
        }

        counts[nearest] += 1;
        for dimension in 0..VECTOR_DIMENSIONS {
            sums[nearest][dimension] += vector[dimension];
        }
    }

    WorkerAccumulation {
        changed,
        counts,
        sums,
    }
}

fn initialize_centroids_kmeans_pp(
    sample_vectors: &[Vector],
    centroid_count: usize,
    rng: &mut SplitMix64,
) -> Vec<Vector> {
    let mut centroids = Vec::with_capacity(centroid_count);
    let first_index = rng.gen_bounded_usize(sample_vectors.len());
    centroids.push(sample_vectors[first_index]);

    let mut nearest_distances = vec![f32::INFINITY; sample_vectors.len()];

    while centroids.len() < centroid_count {
        let newest_centroid = centroids.last().expect("at least one centroid exists");

        for (index, vector) in sample_vectors.iter().enumerate() {
            let distance = l2_squared(vector, newest_centroid);
            if distance < nearest_distances[index] {
                nearest_distances[index] = distance;
            }
        }

        let total_weight = nearest_distances
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .sum::<f32>();

        let next_index = if total_weight <= 0.0 {
            rng.gen_bounded_usize(sample_vectors.len())
        } else {
            let mut target = rng.next_f32() * total_weight;
            let mut selected = sample_vectors.len() - 1;

            for (index, &weight) in nearest_distances.iter().enumerate() {
                if !weight.is_finite() {
                    continue;
                }

                target -= weight;
                if target <= 0.0 {
                    selected = index;
                    break;
                }
            }

            selected
        };

        centroids.push(sample_vectors[next_index]);
    }

    centroids
}

fn build_ivf_index(dataset: &Dataset, model: &KMeansModel) -> PackedIndex {
    let quantized_centroids = model
        .centroids
        .iter()
        .map(|centroid| centroid.map(|value| quantize(value) as f32))
        .collect::<Vec<_>>();
    let mut cluster_sizes = vec![0usize; CLUSTER_COUNT];
    let mut radii = vec![0.0f32; CLUSTER_COUNT];
    for &assignment in &model.assignments {
        cluster_sizes[assignment as usize] += 1;
    }

    for (vector_index, &assignment) in model.assignments.iter().enumerate() {
        let cluster = assignment as usize;
        let mut distance_sq = 0.0f32;

        let cluster_centroids = quantized_centroids[cluster];
        let tx_vector = dataset.vectors[vector_index];

        for dimension in 0..VECTOR_DIMENSIONS {
            let delta = quantize(tx_vector[dimension]) as f32 - cluster_centroids[dimension];
            distance_sq += delta * delta;
        }
        radii[cluster] = radii[cluster].max(distance_sq.sqrt());
    }

    let mut vector_offsets = Vec::with_capacity(CLUSTER_COUNT + 1);
    vector_offsets.push(0usize);
    for &size in &cluster_sizes {
        let next = vector_offsets.last().copied().unwrap_or(0) + size;
        vector_offsets.push(next);
    }

    let mut grouped_indices = vec![0usize; dataset.vectors.len()];
    let mut write_positions = vector_offsets[..CLUSTER_COUNT].to_vec();

    for (vector_index, &assignment) in model.assignments.iter().enumerate() {
        let cluster = assignment as usize;
        let position = write_positions[cluster];
        grouped_indices[position] = vector_index;
        write_positions[cluster] += 1;
    }

    let mut block_offsets = Vec::with_capacity(CLUSTER_COUNT + 1);
    block_offsets.push(0u32);
    for &size in &cluster_sizes {
        let blocks = size.div_ceil(BLOCK_WIDTH) as u32;
        let next = block_offsets.last().copied().unwrap_or(0) + blocks;
        block_offsets.push(next);
    }

    let total_blocks = block_offsets.last().copied().unwrap_or(0) as usize;
    let mut packed_labels = vec![PADDED_LABEL_VALUE; total_blocks * BLOCK_WIDTH];
    let mut quantized_blocks =
        vec![PADDED_QUANTIZED_VALUE; total_blocks * VECTOR_DIMENSIONS * BLOCK_WIDTH];

    for cluster in 0..CLUSTER_COUNT {
        let cluster_start = vector_offsets[cluster];
        let cluster_end = vector_offsets[cluster + 1];
        let block_start = block_offsets[cluster] as usize;

        for (local_index, &vector_index) in grouped_indices[cluster_start..cluster_end]
            .iter()
            .enumerate()
        {
            let block_index = block_start + (local_index / BLOCK_WIDTH);
            let lane = local_index % BLOCK_WIDTH;
            let label_offset = block_index * BLOCK_WIDTH + lane;

            packed_labels[label_offset] = dataset.labels[vector_index];

            for dimension in 0..VECTOR_DIMENSIONS {
                let packed_offset =
                    block_index * VECTOR_DIMENSIONS * BLOCK_WIDTH + dimension * BLOCK_WIDTH + lane;
                quantized_blocks[packed_offset] =
                    quantize(dataset.vectors[vector_index][dimension]);
            }
        }
    }

    PackedIndex {
        centroids_transposed: transpose_centroids(&quantized_centroids),
        radii,
        block_offsets,
        labels: packed_labels,
        quantized_blocks,
    }
}

fn transpose_centroids(centroids: &[Vector]) -> Vec<f32> {
    let mut transposed = Vec::with_capacity(VECTOR_DIMENSIONS * centroids.len());

    for dimension in 0..VECTOR_DIMENSIONS {
        for centroid in centroids {
            transposed.push(centroid[dimension]);
        }
    }

    transposed
}

fn write_index_file(path: &Path, vector_count: u64, index: &PackedIndex) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "failed to create output directory {}: {error}",
                    parent.display()
                ),
            )
        })?;
    }

    let temp_path = temporary_output_path(path);
    let file = File::create(&temp_path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("failed to create {}: {error}", temp_path.display()),
        )
    })?;

    let mut writer = BufWriter::new(file);

    writer.write_all(b"IVF2")?;
    writer.write_all(&(vector_count as u32).to_le_bytes())?;
    writer.write_all(&(CLUSTER_COUNT as u32).to_le_bytes())?;
    writer.write_all(&(VECTOR_DIMENSIONS as u32).to_le_bytes())?;

    for &value in &index.centroids_transposed {
        writer.write_all(&value.to_le_bytes())?;
    }

    for &value in &index.radii {
        writer.write_all(&value.to_le_bytes())?;
    }

    for &offset in &index.block_offsets {
        writer.write_all(&offset.to_le_bytes())?;
    }

    writer.write_all(&index.labels)?;

    for &value in &index.quantized_blocks {
        writer.write_all(&value.to_le_bytes())?;
    }

    writer.flush()?;

    fs::rename(&temp_path, path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "failed to publish index {} from {}: {error}",
                path.display(),
                temp_path.display()
            ),
        )
    })
}

fn temporary_output_path(path: &Path) -> PathBuf {
    let pid = std::process::id();
    let file_name = format!(
        ".{}.tmp.{pid}",
        path.file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("index.bin")
    );

    path.with_file_name(file_name)
}

fn nearest_centroid(vector: &Vector, centroids: &[Vector]) -> usize {
    let mut best_index = 0usize;
    let mut best_distance = f32::INFINITY;

    for (index, centroid) in centroids.iter().enumerate() {
        let distance = l2_squared(vector, centroid);
        if distance < best_distance {
            best_distance = distance;
            best_index = index;
        }
    }

    best_index
}

fn l2_squared(left: &Vector, right: &Vector) -> f32 {
    let mut total = 0.0f32;

    for dimension in 0..VECTOR_DIMENSIONS {
        let delta = left[dimension] - right[dimension];
        total += delta * delta;
    }

    total
}

fn quantize(value: f32) -> i16 {
    let scaled = (value * QUANTIZATION_SCALE).round();
    scaled.clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn reservoir_sample_indices(len: usize, sample_size: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let mut sample = (0..sample_size).collect::<Vec<_>>();

    for index in sample_size..len {
        let replacement = rng.gen_bounded_usize(index + 1);
        if replacement < sample_size {
            sample[replacement] = index;
        }
    }

    sample
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn seeded(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        let value = (self.next_u64() >> 40) as u32;
        value as f32 / ((1u32 << 24) as f32)
    }

    fn gen_bounded_usize(&mut self, upper_bound: usize) -> usize {
        debug_assert!(upper_bound > 0);

        if upper_bound <= 1 {
            return 0;
        }

        (self.next_u64() % upper_bound as u64) as usize
    }
}
