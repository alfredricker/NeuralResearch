use rand::{Rng, RngExt};

// Vose's alias method for exactly 256 bins.
// Build: O(256). Sample: O(1) — two random values, two array lookups.
// 256 bins means index selection is rng.random::<u8>() with zero modulo bias.
fn build_alias_table(weights: [f32; 256]) -> ([f32; 256], [u8; 256]) {
    let total: f32 = weights.iter().sum();
    let mut scaled = [0.0f32; 256];
    for i in 0..256 {
        scaled[i] = weights[i] * 256.0 / total;
    }

    let mut prob  = [0.0f32; 256];
    let mut alias = [0u8;   256];
    let mut small: Vec<usize> = Vec::with_capacity(256);
    let mut large: Vec<usize> = Vec::with_capacity(256);

    for i in 0..256 {
        if scaled[i] < 1.0 { small.push(i); } else { large.push(i); }
    }

    while !small.is_empty() && !large.is_empty() {
        let l = small.pop().unwrap();
        let g = *large.last().unwrap();
        prob[l]  = scaled[l];
        alias[l] = g as u8;
        scaled[g] += scaled[l] - 1.0;
        if scaled[g] < 1.0 { large.pop(); small.push(g); }
    }

    // Any residual entries (floating-point rounding) get probability 1
    for &i in large.iter() { prob[i] = 1.0; }
    for &i in small.iter() { prob[i] = 1.0; }

    (prob, alias)
}

// Precomputed O(1) sampler for i8 values drawn from a discretized normal.
// Construct once per NeuronConfig field; sample many times cheaply.
pub struct SamplerI8 {
    prob:  [f32; 256],
    alias: [u8;  256],
}

impl SamplerI8 {
    pub fn new(mean: i8, std: u8) -> Self {
        let m = mean as f32;
        let mut weights = [0.0f32; 256];
        for (i, w) in weights.iter_mut().enumerate() {
            // bin i maps to i8 value (i - 128): bin 0 → -128, bin 255 → 127
            let x = i as f32 - 128.0;
            *w = if std == 0 {
                if x == m { 1.0 } else { 0.0 }
            } else {
                let z = (x - m) / std as f32;
                (-0.5 * z * z).exp()
            };
        }
        let (prob, alias) = build_alias_table(weights);
        Self { prob, alias }
    }

    pub fn sample(&self, rng: &mut impl Rng) -> i8 {
        let i = rng.random::<u8>() as usize;
        let u = rng.random::<f32>();
        let bin = if u < self.prob[i] { i } else { self.alias[i] as usize };
        (bin as i16 - 128) as i8
    }
}

// Precomputed O(1) sampler for u8 values drawn from a discretized normal.
pub struct SamplerU8 {
    prob:  [f32; 256],
    alias: [u8;  256],
}

impl SamplerU8 {
    pub fn new(mean: u8, std: u8) -> Self {
        let m = mean as f32;
        let mut weights = [0.0f32; 256];
        for (i, w) in weights.iter_mut().enumerate() {
            let x = i as f32;
            *w = if std == 0 {
                if x == m { 1.0 } else { 0.0 }
            } else {
                let z = (x - m) / std as f32;
                (-0.5 * z * z).exp()
            };
        }
        let (prob, alias) = build_alias_table(weights);
        Self { prob, alias }
    }

    pub fn sample(&self, rng: &mut impl Rng) -> u8 {
        let i = rng.random::<u8>() as usize;
        let u = rng.random::<f32>();
        if u < self.prob[i] { i as u8 } else { self.alias[i] }
    }
}

pub fn sample_i8_uniform(lo: i8, hi: i8, rng: &mut impl RngExt) -> i8 {
    rng.random_range(lo..=hi)
}

pub fn sample_u8_uniform(lo: u8, hi: u8, rng: &mut impl RngExt) -> u8 {
    rng.random_range(lo..=hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    const N: usize = 2000;
    const SEED: u64 = 42;

    fn stats(samples: &[f32]) -> (f32, f32) {
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let std  = (samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / samples.len() as f32).sqrt();
        (mean, std)
    }

    // --- SamplerI8 ---

    #[test]
    fn i8_std_zero_returns_mean() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let s = SamplerI8::new(60, 0);
        for _ in 0..100 { assert_eq!(s.sample(&mut rng), 60); }
    }

    #[test]
    fn i8_std_zero_negative_mean() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let s = SamplerI8::new(-10, 0);
        assert_eq!(s.sample(&mut rng), -10);
    }

    #[test]
    fn i8_mean_and_std_within_tolerance() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let s = SamplerI8::new(60, 8);
        let samples: Vec<f32> = (0..N).map(|_| s.sample(&mut rng) as f32).collect();
        let (obs_mean, obs_std) = stats(&samples);
        assert!((obs_mean - 60.0).abs() < 2.0, "mean {obs_mean:.2} not near 60");
        assert!((obs_std  -  8.0).abs() < 2.0, "std {obs_std:.2} not near 8");
    }

    // visual_mnist dendrite constant: mean=60, std=8 — all values should be positive
    #[test]
    fn i8_visual_mnist_dendrite_constant_all_positive() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let s = SamplerI8::new(60, 8);
        assert!((0..N).all(|_| s.sample(&mut rng) > 0));
    }

    #[test]
    fn i8_stays_in_type_range() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let s = SamplerI8::new(0, 60);
        for _ in 0..N {
            let v = s.sample(&mut rng);
            assert!(v >= i8::MIN && v <= i8::MAX);
        }
    }

    // --- SamplerU8 ---

    #[test]
    fn u8_std_zero_returns_mean() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let s = SamplerU8::new(128, 0);
        for _ in 0..100 { assert_eq!(s.sample(&mut rng), 128); }
    }

    #[test]
    fn u8_mean_and_std_within_tolerance() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let s = SamplerU8::new(128, 30);
        let samples: Vec<f32> = (0..N).map(|_| s.sample(&mut rng) as f32).collect();
        let (obs_mean, obs_std) = stats(&samples);
        assert!((obs_mean - 128.0).abs() < 3.0, "mean {obs_mean:.2} not near 128");
        assert!((obs_std  -  30.0).abs() < 3.0, "std {obs_std:.2} not near 30");
    }

    // visual_mnist synapse x: mean=128, std=50 — values should span most of [0, 255]
    #[test]
    fn u8_visual_mnist_synapse_x_spans_range() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let s = SamplerU8::new(128, 50);
        let samples: Vec<u8> = (0..N).map(|_| s.sample(&mut rng)).collect();
        let min = *samples.iter().min().unwrap();
        let max = *samples.iter().max().unwrap();
        assert!(min < 50,  "min={min} unexpectedly high");
        assert!(max > 200, "max={max} unexpectedly low");
    }

    // --- sample_i8_uniform ---

    #[test]
    fn i8_uniform_stays_in_range() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        for _ in 0..N {
            let v = sample_i8_uniform(-8, 8, &mut rng);
            assert!(v >= -8 && v <= 8);
        }
    }

    #[test]
    fn i8_uniform_mean_near_midpoint() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let mean = (0..N)
            .map(|_| sample_i8_uniform(-8, 8, &mut rng) as f32)
            .sum::<f32>() / N as f32;
        assert!(mean.abs() < 1.0, "uniform mean {mean:.2} not near 0");
    }

    // --- sample_u8_uniform ---

    #[test]
    fn u8_uniform_stays_in_range() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        for _ in 0..N { assert!(sample_u8_uniform(0, 8, &mut rng) <= 8); }
    }

    #[test]
    fn u8_uniform_mean_near_midpoint() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let mean = (0..N)
            .map(|_| sample_u8_uniform(0, 8, &mut rng) as f32)
            .sum::<f32>() / N as f32;
        assert!((mean - 4.0).abs() < 0.5, "uniform mean {mean:.2} not near 4");
    }
}
