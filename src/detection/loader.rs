use std::{
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use flate2::read::GzDecoder;

use super::*;

pub fn load_refs(resources_dir: &Path) -> Result<Vec<StoredReference>, FraudEngineError> {
    let compressed_path = resources_dir.join("references.json.gz");

    let file = File::open(&compressed_path).map_err(|error| {
        FraudEngineError::Load(format!(
            "failed to open {}: {error}",
            compressed_path.display()
        ))
    })?;

    let mut reader = BufReader::new(GzDecoder::new(file));
    let mut json = Vec::new();
    reader.read_to_end(&mut json).map_err(|error| {
        FraudEngineError::Load(format!(
            "failed to decompress {}: {error}",
            compressed_path.display()
        ))
    })?;

    let raw_references: Vec<RawReferenceEntry> = sonic_rs::from_slice(&json).map_err(|error| {
        FraudEngineError::Load(format!(
            "failed to parse {}: {error}",
            compressed_path.display()
        ))
    })?;

    Ok(raw_references
        .into_iter()
        .map(StoredReference::from)
        .collect())
}

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

    let mut reader = BufReader::new(file);
    let mut json = Vec::new();
    reader.read_to_end(&mut json).map_err(|error| {
        FraudEngineError::Load(format!("failed to read {}: {error}", path.display()))
    })?;

    sonic_rs::from_slice(&json).map_err(|error| {
        FraudEngineError::Load(format!("failed to parse {}: {error}", path.display()))
    })
}

#[derive(Debug, Clone, Copy)]
pub struct ParsedTimestamp {
    pub unix_seconds: i64,
    pub hour: u8,
    pub weekday_monday0: u8,
}

pub fn parse_utc_timestamp(value: &str) -> Result<ParsedTimestamp, FraudEngineError> {
    let bytes = value.as_bytes();
    if bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
    {
        return Err(FraudEngineError::InvalidRequest(format!(
            "invalid timestamp '{value}': invalid format"
        )));
    }

    let year = parse_4digits(&bytes[0..4], value)? as i32;
    let month = parse_2digits(&bytes[5..7], value)? as u32;
    let day = parse_2digits(&bytes[8..10], value)? as u32;
    let hour = parse_2digits(&bytes[11..13], value)? as u32;
    let minute = parse_2digits(&bytes[14..16], value)? as u32;
    let second = parse_2digits(&bytes[17..19], value)? as u32;

    validate_date_components(year, month, day, hour, minute, second, value)?;

    let days = days_from_civil(year, month, day);
    let unix_seconds = days * 86_400 + hour as i64 * 3_600 + minute as i64 * 60 + second as i64;
    let weekday_monday0 = ((days + 3).rem_euclid(7)) as u8;

    Ok(ParsedTimestamp {
        unix_seconds,
        hour: hour as u8,
        weekday_monday0,
    })
}

impl From<RawReferenceEntry> for StoredReference {
    fn from(value: RawReferenceEntry) -> Self {
        Self {
            quantized_vector: QuantizedVector::from_vector(value.vector),
            label: value.label,
        }
    }
}

fn parse_2digits(bytes: &[u8], original: &str) -> Result<u16, FraudEngineError> {
    if !bytes.iter().all(u8::is_ascii_digit) {
        return Err(FraudEngineError::InvalidRequest(format!(
            "invalid timestamp '{original}': invalid digits"
        )));
    }

    Ok(((bytes[0] - b'0') as u16) * 10 + (bytes[1] - b'0') as u16)
}

fn parse_4digits(bytes: &[u8], original: &str) -> Result<u16, FraudEngineError> {
    if !bytes.iter().all(u8::is_ascii_digit) {
        return Err(FraudEngineError::InvalidRequest(format!(
            "invalid timestamp '{original}': invalid digits"
        )));
    }

    Ok(((bytes[0] - b'0') as u16) * 1_000
        + ((bytes[1] - b'0') as u16) * 100
        + ((bytes[2] - b'0') as u16) * 10
        + (bytes[3] - b'0') as u16)
}

fn validate_date_components(
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
    original: &str,
) -> Result<(), FraudEngineError> {
    if !(1..=12).contains(&month) || day == 0 || hour > 23 || minute > 59 || second > 59 {
        return Err(FraudEngineError::InvalidRequest(format!(
            "invalid timestamp '{original}': out of range"
        )));
    }

    let max_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => unreachable!(),
    };

    if day > max_day {
        return Err(FraudEngineError::InvalidRequest(format!(
            "invalid timestamp '{original}': out of range"
        )));
    }

    Ok(())
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let year = year - if month <= 2 { 1 } else { 0 };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let month_prime = month as i32 + if month > 2 { -3 } else { 9 };
    let doy = (153 * month_prime + 2) / 5 + day as i32 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era as i64 * 146_097 + doe as i64 - 719_468
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
