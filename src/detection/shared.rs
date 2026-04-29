use std::{
    mem::{align_of, size_of},
    ptr, slice,
};

use super::*;

const EMBEDDED_DATASET_BYTES: &[u8] = include_bytes!("../../index.mmap");
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

#[derive(Debug, Clone, Copy)]
pub struct EmbeddedDataset {
    bytes: &'static [u8],
    header: DatasetHeader,
}

pub fn load_embedded_dataset() -> Result<EmbeddedDataset, FraudEngineError> {
    let header = validate_header_bytes(EMBEDDED_DATASET_BYTES)?;
    if header.clustering_enabled == 0 {
        return Err(FraudEngineError::Load(
            "embedded dataset must be a centroid-reduced artifact".to_owned(),
        ));
    }

    let dataset = EmbeddedDataset {
        bytes: EMBEDDED_DATASET_BYTES,
        header,
    };
    validate_dataset_contents(dataset.len(), dataset.labels())?;
    Ok(dataset)
}

impl EmbeddedDataset {
    pub fn len(&self) -> usize {
        self.header.reference_count as usize
    }

    pub fn vector(&self, index: usize) -> &QuantizedVector {
        &self.vectors()[index]
    }

    pub fn label(&self, index: usize) -> ReferenceLabel {
        ReferenceLabel::from_storage_byte(self.labels()[index])
            .expect("embedded dataset validated labels before use")
    }

    fn vectors(&self) -> &[QuantizedVector] {
        slice_from_offset(
            self.bytes,
            self.header.vectors_offset,
            self.header.reference_count as usize,
        )
    }

    fn labels(&self) -> &[u8] {
        slice_from_offset(
            self.bytes,
            self.header.labels_offset,
            self.header.reference_count as usize,
        )
    }
}

fn validate_header_bytes(bytes: &[u8]) -> Result<DatasetHeader, FraudEngineError> {
    if bytes.len() < size_of::<DatasetHeader>() {
        return Err(FraudEngineError::Load(
            "embedded dataset file is smaller than its header".to_owned(),
        ));
    }

    let header = unsafe { ptr::read_unaligned(bytes.as_ptr().cast::<DatasetHeader>()) };

    if header.magic != DATASET_MAGIC
        || header.version != DATASET_VERSION
        || header.logical_dimensions != VECTOR_DIMENSIONS as u32
        || header.stored_dimensions != STORED_VECTOR_DIMENSIONS as u32
        || header.quantization_scale != QUANTIZATION_SCALE
        || header.file_len as usize != bytes.len()
    {
        return Err(FraudEngineError::Load(
            "embedded dataset header does not match expected layout".to_owned(),
        ));
    }

    validate_region::<QuantizedVector>(&header, header.vectors_offset, header.reference_count)?;
    validate_region::<u8>(&header, header.labels_offset, header.reference_count)?;

    Ok(header)
}

fn validate_dataset_contents(len: usize, labels: &[u8]) -> Result<(), FraudEngineError> {
    if len < K_NEIGHBORS {
        return Err(FraudEngineError::Load(format!(
            "embedded dataset must contain at least {K_NEIGHBORS} vectors, found {}",
            len
        )));
    }

    if labels
        .iter()
        .any(|label| ReferenceLabel::from_storage_byte(*label).is_none())
    {
        return Err(FraudEngineError::Load(
            "embedded dataset contains an invalid label value".to_owned(),
        ));
    }

    Ok(())
}

fn validate_region<T>(
    header: &DatasetHeader,
    offset: u64,
    len: u64,
) -> Result<(), FraudEngineError> {
    let alignment = align_of::<T>() as u64;
    let element_size = size_of::<T>() as u64;

    if !offset.is_multiple_of(alignment) {
        return Err(FraudEngineError::Load(
            "embedded dataset contains a misaligned section".to_owned(),
        ));
    }

    let byte_len = len.checked_mul(element_size).ok_or_else(|| {
        FraudEngineError::Load("embedded dataset section length overflow".to_owned())
    })?;
    let end = offset.checked_add(byte_len).ok_or_else(|| {
        FraudEngineError::Load("embedded dataset section offset overflow".to_owned())
    })?;

    if end > header.file_len {
        return Err(FraudEngineError::Load(
            "embedded dataset section exceeds file length".to_owned(),
        ));
    }

    Ok(())
}

fn slice_from_offset<T>(bytes: &[u8], offset: u64, len: usize) -> &[T] {
    let start = offset as usize;
    let ptr = unsafe { bytes.as_ptr().add(start).cast::<T>() };
    unsafe { slice::from_raw_parts(ptr, len) }
}
