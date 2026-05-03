use super::*;

#[inline]
pub fn normalize_amount_vs_avg(amount: f32, customer_average: f32, scaling_ratio: f32) -> f32 {
    if customer_average <= 0.0 {
        return if amount <= 0.0 { 0.0 } else { 1.0 };
    }

    clamp_unit((amount / customer_average) / scaling_ratio)
}

#[inline]
pub fn clamp_ratio(value: f32, max_value: f32) -> f32 {
    if max_value <= 0.0 {
        return 0.0;
    }

    clamp_unit(value / max_value)
}

#[inline]
fn clamp_unit(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

#[inline]
pub fn bool_to_unit(value: bool) -> f32 {
    if value { 1.0 } else { 0.0 }
}

#[inline]
pub fn l2_squared(left: &[f32; VECTOR_DIMENSIONS], right: &[f32; VECTOR_DIMENSIONS]) -> f32 {
    let mut total = 0.0f32;

    for dimension in 0..VECTOR_DIMENSIONS {
        let delta = left[dimension] - right[dimension];
        total += delta * delta;
    }

    total
}

#[inline]
pub fn quantize(value: f32) -> i16 {
    let scaled = (value * QUANTIZATION_SCALE).round();
    scaled.clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

#[inline]
pub fn quantized_as_f32(value: f32) -> f32 {
    quantize(value) as f32
}
