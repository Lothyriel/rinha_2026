use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

use chrono::{DateTime, Utc};

use super::*;

pub fn load_example_refs(resources_dir: &Path) -> Result<Vec<StoredReference>, FraudEngineError> {
    let example_path = resources_dir.join("example-references.json");
    let raw_references: Vec<RawReferenceEntry> = load_json_file(example_path)?;

    Ok(raw_references
        .into_iter()
        .map(StoredReference::from)
        .collect())
}

pub fn load_json_file<T: serde::de::DeserializeOwned>(
    path: PathBuf,
) -> Result<T, FraudEngineError> {
    let file = File::open(&path).map_err(|error| {
        FraudEngineError::Load(format!("failed to open {}: {error}", path.display()))
    })?;

    let reader = BufReader::new(file);

    serde_json::from_reader(reader).map_err(|error| {
        FraudEngineError::Load(format!("failed to parse {}: {error}", path.display()))
    })
}

pub fn parse_utc_timestamp(value: &str) -> Result<DateTime<Utc>, FraudEngineError> {
    DateTime::parse_from_rfc3339(value)
        .map(|timestamp| timestamp.with_timezone(&Utc))
        .map_err(|error| {
            FraudEngineError::InvalidRequest(format!("invalid timestamp '{value}': {error}"))
        })
}

pub fn load_prebuilt_index() -> Result<IvfIndex, FraudEngineError> {
    let bytes = include_bytes!("../../index.bin");

    parse_prebuilt_index(bytes)
}

fn parse_prebuilt_index(bytes: &[u8]) -> Result<IvfIndex, FraudEngineError> {
    let mut cursor = 0usize;

    let magic = read_exact::<4>(bytes, &mut cursor, "magic")?;
    if magic != PREBUILT_INDEX_MAGIC {
        return Err(FraudEngineError::Load(format!(
            "prebuilt index has unsupported magic {:?}",
            String::from_utf8_lossy(&magic)
        )));
    }

    let reference_count = read_u32(bytes, &mut cursor, "reference count")? as usize;
    let cluster_count = read_u32(bytes, &mut cursor, "cluster count")? as usize;
    let dimensions = read_u32(bytes, &mut cursor, "dimensions")? as usize;

    if dimensions != VECTOR_DIMENSIONS {
        return Err(FraudEngineError::Load(format!(
            "prebuilt index expected {} dimensions, found {}",
            VECTOR_DIMENSIONS, dimensions
        )));
    }

    let centroid_len = cluster_count
        .checked_mul(VECTOR_DIMENSIONS)
        .ok_or_else(|| {
            FraudEngineError::Load("prebuilt index centroid length overflow".to_owned())
        })?;
    let centroids_transposed = read_f32_vec(bytes, &mut cursor, centroid_len, "centroids")?;
    let radii = read_f32_vec(bytes, &mut cursor, cluster_count, "radii")?;
    let block_offsets = read_u32_vec(bytes, &mut cursor, cluster_count + 1, "block offsets")?;
    let block_count = block_offsets.last().copied().unwrap_or(0) as usize;
    let labels = read_u8_vec(
        bytes,
        &mut cursor,
        block_count.checked_mul(BLOCK_WIDTH).ok_or_else(|| {
            FraudEngineError::Load("prebuilt index labels length overflow".to_owned())
        })?,
        "labels",
    )?;
    let quantized_blocks = read_i16_vec(
        bytes,
        &mut cursor,
        block_count
            .checked_mul(VECTOR_DIMENSIONS)
            .and_then(|value| value.checked_mul(BLOCK_WIDTH))
            .ok_or_else(|| {
                FraudEngineError::Load("prebuilt index blocks length overflow".to_owned())
            })?,
        "quantized blocks",
    )?;

    if cursor != bytes.len() {
        return Err(FraudEngineError::Load(format!(
            "prebuilt index has {} trailing bytes",
            bytes.len() - cursor
        )));
    }

    Ok(IvfIndex {
        reference_count,
        centroids_transposed,
        radii,
        block_offsets,
        labels,
        quantized_blocks,
    })
}

fn read_exact<const N: usize>(
    bytes: &[u8],
    cursor: &mut usize,
    field: &str,
) -> Result<[u8; N], FraudEngineError> {
    let end = cursor.checked_add(N).ok_or_else(|| {
        FraudEngineError::Load(format!("prebuilt index overflow while reading {field}",))
    })?;
    let slice = bytes.get(*cursor..end).ok_or_else(|| {
        FraudEngineError::Load(format!("prebuilt index truncated while reading {field}",))
    })?;
    *cursor = end;
    slice
        .try_into()
        .map_err(|_| FraudEngineError::Load(format!("prebuilt index failed to decode {field}",)))
}

fn read_u32(bytes: &[u8], cursor: &mut usize, field: &str) -> Result<u32, FraudEngineError> {
    Ok(u32::from_le_bytes(read_exact(bytes, cursor, field)?))
}

fn read_f32_vec(
    bytes: &[u8],
    cursor: &mut usize,
    len: usize,
    field: &str,
) -> Result<Vec<f32>, FraudEngineError> {
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(f32::from_le_bytes(read_exact(bytes, cursor, field)?));
    }
    Ok(values)
}

fn read_u32_vec(
    bytes: &[u8],
    cursor: &mut usize,
    len: usize,
    field: &str,
) -> Result<Vec<u32>, FraudEngineError> {
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(u32::from_le_bytes(read_exact(bytes, cursor, field)?));
    }
    Ok(values)
}

fn read_u8_vec(
    bytes: &[u8],
    cursor: &mut usize,
    len: usize,
    field: &str,
) -> Result<Vec<u8>, FraudEngineError> {
    let end = cursor.checked_add(len).ok_or_else(|| {
        FraudEngineError::Load(format!("prebuilt index  overflow while reading {field}",))
    })?;
    let slice = bytes.get(*cursor..end).ok_or_else(|| {
        FraudEngineError::Load(format!("prebuilt index truncated while reading {field}",))
    })?;
    *cursor = end;
    Ok(slice.to_vec())
}

fn read_i16_vec(
    bytes: &[u8],
    cursor: &mut usize,
    len: usize,
    field: &str,
) -> Result<Vec<i16>, FraudEngineError> {
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(i16::from_le_bytes(read_exact(bytes, cursor, field)?));
    }
    Ok(values)
}

impl From<RawReferenceEntry> for StoredReference {
    fn from(value: RawReferenceEntry) -> Self {
        Self {
            vector: value.vector,
            label: value.label,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn loads_official_reference() {
        let engine =
            FraudEngine::load(Path::new("./spec/resources")).expect("spec resources should load");

        assert!(engine.reference_count() >= 100_000);
    }
}
