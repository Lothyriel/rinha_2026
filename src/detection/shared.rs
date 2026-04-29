use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    mem::{align_of, size_of},
    path::{Path, PathBuf},
    ptr,
    slice,
    thread,
    time::Duration,
};

use memmap2::{Mmap, MmapMut, MmapOptions};

use super::*;

const SHARED_DATASET_MAGIC: [u8; 8] = *b"R26MMAP\0";
const SHARED_DATASET_VERSION: u32 = 1;
const SHARED_DATASET_WAIT_RETRIES: usize = 1_200;
const SHARED_DATASET_WAIT_INTERVAL: Duration = Duration::from_millis(100);

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct SharedDatasetHeader {
    magic: [u8; 8],
    version: u32,
    leaf_size: u32,
    reference_count: u64,
    node_count: u64,
    index_count: u64,
    vectors_offset: u64,
    labels_offset: u64,
    nodes_offset: u64,
    indices_offset: u64,
    file_len: u64,
}

#[derive(Debug)]
pub struct MappedDataset {
    mmap: Mmap,
    header: SharedDatasetHeader,
}

pub fn load_or_create_mapped_dataset(
    resources_dir: &Path,
    mmap_path: &Path,
    leaf_size: usize,
) -> Result<MappedDataset, FraudEngineError> {
    let actor = current_actor();

    if let Some(dataset) = try_open_mapped_dataset(mmap_path, leaf_size)? {
        tracing::info!(
            actor = %actor,
            mmap_path = %mmap_path.display(),
            "shared dataset mmap already available"
        );
        return Ok(dataset);
    }

    let lock_path = lock_path_for(mmap_path);

    tracing::info!(
        actor = %actor,
        mmap_path = %mmap_path.display(),
        lock_path = %lock_path.display(),
        "shared dataset mmap not ready; coordinating creation"
    );

    for _ in 0..SHARED_DATASET_WAIT_RETRIES {
        if let Some(dataset) = try_open_mapped_dataset(mmap_path, leaf_size)? {
            tracing::info!(
                actor = %actor,
                mmap_path = %mmap_path.display(),
                "shared dataset mmap became available"
            );
            return Ok(dataset);
        }

        if let Some(_lock) = try_acquire_init_lock(&lock_path)? {
            tracing::info!(
                actor = %actor,
                mmap_path = %mmap_path.display(),
                lock_path = %lock_path.display(),
                "this process will create the shared dataset mmap file"
            );
            build_shared_dataset_file(resources_dir, mmap_path, leaf_size)?;

            if let Some(dataset) = try_open_mapped_dataset(mmap_path, leaf_size)? {
                tracing::info!(
                    actor = %actor,
                    mmap_path = %mmap_path.display(),
                    "shared dataset mmap creation completed successfully"
                );
                return Ok(dataset);
            }

            return Err(FraudEngineError::Load(format!(
                "shared dataset {} was built but could not be reopened",
                mmap_path.display()
            )));
        }

        let lock_owner = read_init_lock_owner(&lock_path)
            .unwrap_or_else(|| "unknown actor".to_owned());

        tracing::info!(
            actor = %actor,
            mmap_path = %mmap_path.display(),
            lock_path = %lock_path.display(),
            lock_owner,
            wait_ms = SHARED_DATASET_WAIT_INTERVAL.as_millis(),
            "shared dataset mmap init lock busy; waiting for creator to finish"
        );

        thread::sleep(SHARED_DATASET_WAIT_INTERVAL);
    }

    Err(FraudEngineError::Load(format!(
        "timed out waiting for shared dataset {} to become ready",
        mmap_path.display()
    )))
}

impl MappedDataset {
    pub fn len(&self) -> usize {
        self.header.reference_count as usize
    }

    pub fn vector(&self, index: usize) -> &Vec16 {
        &self.vectors()[index]
    }

    pub fn label(&self, index: usize) -> ReferenceLabel {
        ReferenceLabel::from_storage_byte(self.labels()[index])
            .expect("shared dataset validated labels before use")
    }

    pub fn nodes(&self) -> &[VpNode] {
        self.slice_from_offset(self.header.nodes_offset, self.header.node_count as usize)
    }

    pub fn indices(&self) -> &[u32] {
        self.slice_from_offset(self.header.indices_offset, self.header.index_count as usize)
    }

    fn vectors(&self) -> &[Vec16] {
        self.slice_from_offset(self.header.vectors_offset, self.header.reference_count as usize)
    }

    fn labels(&self) -> &[u8] {
        self.slice_from_offset(self.header.labels_offset, self.header.reference_count as usize)
    }

    fn slice_from_offset<T>(&self, offset: u64, len: usize) -> &[T] {
        let start = offset as usize;
        let ptr = unsafe { self.mmap.as_ptr().add(start).cast::<T>() };

        unsafe { slice::from_raw_parts(ptr, len) }
    }
}

fn try_open_mapped_dataset(
    mmap_path: &Path,
    leaf_size: usize,
) -> Result<Option<MappedDataset>, FraudEngineError> {
    let file = match OpenOptions::new().read(true).open(mmap_path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(FraudEngineError::Load(format!(
                "failed to open shared dataset {}: {error}",
                mmap_path.display()
            )))
        }
    };

    let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|error| {
        FraudEngineError::Load(format!(
            "failed to mmap shared dataset {}: {error}",
            mmap_path.display()
        ))
    })?;

    let header = match validate_header(&mmap, leaf_size) {
        Ok(header) => header,
        Err(_) => return Ok(None),
    };

    let dataset = MappedDataset { mmap, header };
    validate_dataset_contents(&dataset)?;

    Ok(Some(dataset))
}

fn validate_header(mmap: &Mmap, leaf_size: usize) -> Result<SharedDatasetHeader, FraudEngineError> {
    if mmap.len() < size_of::<SharedDatasetHeader>() {
        return Err(FraudEngineError::Load(
            "shared dataset file is smaller than its header".to_owned(),
        ));
    }

    let header = unsafe { ptr::read_unaligned(mmap.as_ptr().cast::<SharedDatasetHeader>()) };

    if header.magic != SHARED_DATASET_MAGIC
        || header.version != SHARED_DATASET_VERSION
        || header.leaf_size != leaf_size as u32
        || header.file_len as usize != mmap.len()
    {
        return Err(FraudEngineError::Load(
            "shared dataset header does not match expected layout".to_owned(),
        ));
    }

    validate_region::<Vec16>(&header, header.vectors_offset, header.reference_count)?;
    validate_region::<u8>(&header, header.labels_offset, header.reference_count)?;
    validate_region::<VpNode>(&header, header.nodes_offset, header.node_count)?;
    validate_region::<u32>(&header, header.indices_offset, header.index_count)?;

    Ok(header)
}

fn validate_dataset_contents(dataset: &MappedDataset) -> Result<(), FraudEngineError> {
    if dataset.len() < K_NEIGHBORS {
        return Err(FraudEngineError::Load(format!(
            "shared dataset must contain at least {K_NEIGHBORS} vectors, found {}",
            dataset.len()
        )));
    }

    if dataset
        .labels()
        .iter()
        .any(|label| ReferenceLabel::from_storage_byte(*label).is_none())
    {
        return Err(FraudEngineError::Load(
            "shared dataset contains an invalid label value".to_owned(),
        ));
    }

    Ok(())
}

fn validate_region<T>(
    header: &SharedDatasetHeader,
    offset: u64,
    len: u64,
) -> Result<(), FraudEngineError> {
    let alignment = align_of::<T>() as u64;
    let element_size = size_of::<T>() as u64;

    if offset % alignment != 0 {
        return Err(FraudEngineError::Load(
            "shared dataset contains a misaligned section".to_owned(),
        ));
    }

    let byte_len = len
        .checked_mul(element_size)
        .ok_or_else(|| FraudEngineError::Load("shared dataset section length overflow".to_owned()))?;
    let end = offset
        .checked_add(byte_len)
        .ok_or_else(|| FraudEngineError::Load("shared dataset section offset overflow".to_owned()))?;

    if end > header.file_len {
        return Err(FraudEngineError::Load(
            "shared dataset section exceeds file length".to_owned(),
        ));
    }

    Ok(())
}

pub(super) fn build_shared_dataset_file(
    resources_dir: &Path,
    mmap_path: &Path,
    leaf_size: usize,
) -> Result<(), FraudEngineError> {
    tracing::info!(
        actor = %current_actor(),
        mmap_path = %mmap_path.display(),
        resources_dir = %resources_dir.display(),
        leaf_size,
        "building shared dataset mmap file"
    );

    if let Some(parent) = mmap_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            FraudEngineError::Load(format!(
                "failed to create shared dataset directory {}: {error}",
                parent.display()
            ))
        })?;
    }

    let references = loader::load_refs(resources_dir)?;

    if references.len() < K_NEIGHBORS {
        return Err(FraudEngineError::Load(format!(
            "reference dataset must contain at least {K_NEIGHBORS} vectors, found {}",
            references.len()
        )));
    }

    let index = search::build_index(&references, leaf_size);
    let temp_path = temporary_dataset_path(mmap_path);

    write_dataset_file(&temp_path, &references, &index, leaf_size)?;
    fs::rename(&temp_path, mmap_path).map_err(|error| {
        FraudEngineError::Load(format!(
            "failed to publish shared dataset {}: {error}",
            mmap_path.display()
        ))
    })?;

    tracing::info!(
        actor = %current_actor(),
        mmap_path = %mmap_path.display(),
        temp_path = %temp_path.display(),
        "shared dataset mmap file published successfully"
    );

    Ok(())
}

fn write_dataset_file(
    path: &Path,
    references: &[StoredReference],
    index: &ExactSearchIndex,
    leaf_size: usize,
) -> Result<(), FraudEngineError> {
    let vectors_offset = align_up(size_of::<SharedDatasetHeader>() as u64, align_of::<Vec16>() as u64);
    let labels_offset = align_up(
        vectors_offset + byte_len::<Vec16>(references.len())?,
        align_of::<u8>() as u64,
    );
    let nodes_offset = align_up(
        labels_offset + byte_len::<u8>(references.len())?,
        align_of::<VpNode>() as u64,
    );
    let indices_offset = align_up(
        nodes_offset + byte_len::<VpNode>(index.nodes.len())?,
        align_of::<u32>() as u64,
    );
    let file_len = indices_offset + byte_len::<u32>(index.indices.len())?;

    let header = SharedDatasetHeader {
        magic: SHARED_DATASET_MAGIC,
        version: SHARED_DATASET_VERSION,
        leaf_size: leaf_size as u32,
        reference_count: references.len() as u64,
        node_count: index.nodes.len() as u64,
        index_count: index.indices.len() as u64,
        vectors_offset,
        labels_offset,
        nodes_offset,
        indices_offset,
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
                "failed to create shared dataset file {}: {error}",
                path.display()
            ))
        })?;

    file.set_len(file_len).map_err(|error| {
        FraudEngineError::Load(format!(
            "failed to size shared dataset file {}: {error}",
            path.display()
        ))
    })?;

    let mut mmap = unsafe { MmapMut::map_mut(&file) }.map_err(|error| {
        FraudEngineError::Load(format!(
            "failed to mmap shared dataset file {}: {error}",
            path.display()
        ))
    })?;

    write_bytes(&mut mmap, 0, bytes_of(&header));

    let vectors = references.iter().map(|reference| reference.padded_vector).collect::<Vec<_>>();
    write_bytes(&mut mmap, vectors_offset as usize, bytes_of_slice(&vectors));

    let labels = references
        .iter()
        .map(|reference| reference.label.to_storage_byte())
        .collect::<Vec<_>>();
    write_bytes(&mut mmap, labels_offset as usize, &labels);
    write_bytes(
        &mut mmap,
        nodes_offset as usize,
        bytes_of_slice(&index.nodes),
    );
    write_bytes(
        &mut mmap,
        indices_offset as usize,
        bytes_of_slice(&index.indices),
    );

    mmap.flush().map_err(|error| {
        FraudEngineError::Load(format!(
            "failed to flush shared dataset file {}: {error}",
            path.display()
        ))
    })
}

fn bytes_of<T>(value: &T) -> &[u8] {
    unsafe { slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) }
}

fn bytes_of_slice<T>(value: &[T]) -> &[u8] {
    unsafe { slice::from_raw_parts(value.as_ptr().cast::<u8>(), size_of_val(value)) }
}

fn write_bytes(mmap: &mut MmapMut, offset: usize, bytes: &[u8]) {
    mmap[offset..offset + bytes.len()].copy_from_slice(bytes);
}

fn byte_len<T>(len: usize) -> Result<u64, FraudEngineError> {
    (len as u64)
        .checked_mul(size_of::<T>() as u64)
        .ok_or_else(|| FraudEngineError::Load("shared dataset size overflow".to_owned()))
}

fn align_up(offset: u64, alignment: u64) -> u64 {
    let mask = alignment - 1;
    (offset + mask) & !mask
}

fn temporary_dataset_path(path: &Path) -> PathBuf {
    let pid = std::process::id();
    let file_name = format!(
        ".{}.tmp.{pid}",
        path.file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("shared-dataset")
    );

    path.with_file_name(file_name)
}

fn lock_path_for(path: &Path) -> PathBuf {
    let file_name = format!(
        ".{}.lock",
        path.file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("shared-dataset")
    );

    path.with_file_name(file_name)
}

fn try_acquire_init_lock(lock_path: &Path) -> Result<Option<InitLock>, FraudEngineError> {
    let owner = current_actor();

    match OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(lock_path)
    {
        Ok(mut file) => {
            file.write_all(owner.as_bytes()).map_err(|error| {
                let _ = fs::remove_file(lock_path);
                FraudEngineError::Load(format!(
                    "failed to record shared dataset lock owner {}: {error}",
                    lock_path.display()
                ))
            })?;

            file.sync_all().map_err(|error| {
                let _ = fs::remove_file(lock_path);
                FraudEngineError::Load(format!(
                    "failed to persist shared dataset lock owner {}: {error}",
                    lock_path.display()
                ))
            })?;

            tracing::info!(
                actor = %owner,
                lock_path = %lock_path.display(),
                "shared dataset init lock acquired"
            );

            Ok(Some(InitLock {
                path: lock_path.to_path_buf(),
                owner,
            }))
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => Ok(None),
        Err(error) => Err(FraudEngineError::Load(format!(
            "failed to create shared dataset lock {}: {error}",
            lock_path.display()
        ))),
    }
}

struct InitLock {
    path: PathBuf,
    owner: String,
}

impl Drop for InitLock {
    fn drop(&mut self) {
        match fs::remove_file(&self.path) {
            Ok(()) => tracing::info!(
                actor = %self.owner,
                lock_path = %self.path.display(),
                "shared dataset init lock released"
            ),
            Err(error) if error.kind() == io::ErrorKind::NotFound => tracing::info!(
                actor = %self.owner,
                lock_path = %self.path.display(),
                "shared dataset init lock already absent during release"
            ),
            Err(error) => tracing::warn!(
                actor = %self.owner,
                lock_path = %self.path.display(),
                ?error,
                "failed to release shared dataset init lock"
            ),
        }
    }
}

fn current_actor() -> String {
    let pid = std::process::id();
    let thread = thread::current();
    let thread_name = thread.name().unwrap_or("unnamed");

    format!("pid={pid} thread={thread_name}")
}

fn read_init_lock_owner(lock_path: &Path) -> Option<String> {
    fs::read_to_string(lock_path)
        .ok()
        .map(|content| content.trim().to_owned())
        .filter(|content| !content.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_reference(vector: [f32; VECTOR_DIMENSIONS], label: ReferenceLabel) -> StoredReference {
        StoredReference {
            padded_vector: Vec16::from_vector(vector),
            label,
        }
    }

    #[test]
    fn round_trips_shared_dataset_file() {
        let temp_dir = std::env::temp_dir().join(format!(
            "rinha_2026_shared_dataset_test_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("thread")
        ));
        fs::create_dir_all(&temp_dir).expect("temp dir should exist");

        let dataset_path = temp_dir.join("dataset.bin");
        let references = vec![
            test_reference([0.1; VECTOR_DIMENSIONS], ReferenceLabel::Fraud),
            test_reference([0.2; VECTOR_DIMENSIONS], ReferenceLabel::Legit),
            test_reference([0.3; VECTOR_DIMENSIONS], ReferenceLabel::Fraud),
            test_reference([0.4; VECTOR_DIMENSIONS], ReferenceLabel::Legit),
            test_reference([0.5; VECTOR_DIMENSIONS], ReferenceLabel::Fraud),
        ];
        let index = search::build_index(&references, LEAF_SIZE);

        write_dataset_file(&dataset_path, &references, &index, LEAF_SIZE)
            .expect("dataset file should be written");
        let mapped = try_open_mapped_dataset(&dataset_path, LEAF_SIZE)
            .expect("mapped dataset should open")
            .expect("mapped dataset should exist");

        assert_eq!(mapped.len(), references.len());
        assert_eq!(mapped.nodes().len(), index.nodes.len());
        assert_eq!(mapped.indices(), index.indices.as_slice());

        for (index, reference) in references.iter().enumerate() {
            assert_eq!(mapped.vector(index).0, reference.padded_vector.0);
            assert_eq!(mapped.label(index), reference.label);
        }

        let _ = fs::remove_file(dataset_path);
        let _ = fs::remove_dir_all(temp_dir);
    }
}
