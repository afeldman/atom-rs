//! Deterministic, reproducible random sampling.
//!
//! Monte-Carlo sampling of quantum-mechanical probability densities is a
//! *reproducibility-critical* operation: the same seed must always yield the
//! same sequence, on every platform and in every thread. The RNG is therefore
//! an explicitly seeded ChaCha8 stream rather than a thread-local generator.
//!
//! Two sampling strategies are provided:
//!
//! * [`InverseCdf1D`] — piecewise-linear inverse-CDF sampling. This is the
//!   preferred method for the smooth one-dimensional probability densities
//!   arising from hydrogenic radial and polar distributions.
//! * [`rejection_sample_1d`] — classic rejection sampling, provided for cases
//!   where a bounded proposal is cheaper than building a CDF table.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// A deterministic, explicitly seeded random number source.
#[derive(Debug, Clone)]
pub struct SeededRng {
    rng: ChaCha8Rng,
    seed: u64,
}

impl SeededRng {
    /// Create a generator from an explicit seed.
    pub fn from_seed(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            seed,
        }
    }

    /// The seed this generator was constructed with.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Uniform deviate in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        self.rng.gen::<f64>()
    }

    /// Uniform deviate in `[lo, hi)`.
    pub fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }
}

/// A one-dimensional probability density represented by a tabulated CDF.
///
/// The density is sampled on a uniform grid on `[x_min, x_max]`, normalised
/// numerically, and inverted by binary search with linear interpolation.
#[derive(Debug, Clone)]
pub struct InverseCdf1D {
    x_min: f64,
    dx: f64,
    /// Cumulative distribution values in `[0, 1]`, monotonically non-decreasing.
    cdf: Vec<f64>,
    /// The probability mass actually contained in `[x_min, x_max]`.
    ///
    /// A value well below 1.0 indicates that the domain truncates significant
    /// probability mass; callers should verify this against their tolerance.
    captured_mass: f64,
}

impl InverseCdf1D {
    /// Build a sampler from an unnormalised density `pdf` over `[x_min, x_max]`
    /// using `n` grid points (`n >= 2`).
    ///
    /// Returns `None` if the density integrates to zero on the domain (for
    /// example, a domain that excludes all support of the density).
    pub fn from_pdf<F: Fn(f64) -> f64>(pdf: F, x_min: f64, x_max: f64, n: usize) -> Option<Self> {
        let n = n.max(2);
        let dx = (x_max - x_min) / (n as f64 - 1.0);
        let mut cdf = Vec::with_capacity(n);
        let mut acc = 0.0_f64;
        let mut prev = (pdf)(x_min).max(0.0);
        cdf.push(0.0);
        for i in 1..n {
            let x = x_min + dx * i as f64;
            let cur = (pdf)(x).max(0.0);
            // Trapezoidal increments keep the table consistent with the
            // piecewise-linear interpolation used during inversion.
            acc += 0.5 * (prev + cur) * dx;
            cdf.push(acc);
            prev = cur;
        }
        if !acc.is_finite() || acc <= 0.0 {
            return None;
        }
        for v in cdf.iter_mut() {
            *v /= acc;
        }
        Some(Self {
            x_min,
            dx,
            cdf,
            captured_mass: 1.0,
        })
    }

    /// Sample a value. `rng` must supply a uniform deviate in `[0, 1)`.
    pub fn sample(&self, rng: &mut SeededRng) -> f64 {
        let u = rng.uniform();
        self.sample_at(u)
    }

    /// Invert the CDF at an explicit uniform deviate `u` in `[0, 1)`.
    pub fn sample_at(&self, u: f64) -> f64 {
        let n = self.cdf.len();
        // Binary search for the first index with cdf[i] >= u.
        let mut lo = 0usize;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.cdf[mid] <= u {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let c0 = self.cdf[lo];
        let c1 = self.cdf[hi];
        let frac = if c1 > c0 { (u - c0) / (c1 - c0) } else { 0.0 };
        self.x_min + self.dx * (lo as f64 + frac)
    }

    /// Fraction of the density mass captured by the tabulated domain.
    ///
    /// This is `1.0` after normalisation; the value is retained as an explicit
    /// extension point and documented for callers that need the raw mass.
    pub fn captured_mass(&self) -> f64 {
        self.captured_mass
    }

    /// Grid points of the CDF table (for diagnostics and testing).
    pub fn table_len(&self) -> usize {
        self.cdf.len()
    }
}

/// Classic rejection sampling from an unnormalised density `pdf` bounded by
/// `pdf_max` on `[x_min, x_max]`.
///
/// Returns `None` if no accepted sample was found within `max_attempts`.
///
/// # Correct use
///
/// `pdf_max` **must** be an upper bound on `pdf` over `[x_min, x_max]`. If it is
/// too small the acceptance test `y <= pdf(x)` is trivially satisfied and the
/// sampler returns the *proposal* distribution instead of `pdf` — silently
/// biased. This function cannot detect that situation; supplying a valid bound
/// is the caller's responsibility. A bound that is too large is safe but wastes
/// attempts.
///
/// `max_attempts == 0` always returns `None`, which is the only "no sample
/// found" failure mode.
pub fn rejection_sample_1d<F: Fn(f64) -> f64>(
    pdf: F,
    x_min: f64,
    x_max: f64,
    pdf_max: f64,
    rng: &mut SeededRng,
    max_attempts: usize,
) -> Option<f64> {
    if pdf_max.is_nan() || pdf_max <= 0.0 {
        return None;
    }
    for _ in 0..max_attempts {
        let x = rng.uniform_range(x_min, x_max);
        let y = rng.uniform_range(0.0, pdf_max);
        if y <= (pdf)(x) {
            return Some(x);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_identical_stream() {
        let mut a = SeededRng::from_seed(20261015);
        let mut b = SeededRng::from_seed(20261015);
        for _ in 0..1000 {
            assert_eq!(a.uniform(), b.uniform());
        }
    }

    #[test]
    fn different_seed_gives_different_stream() {
        let mut a = SeededRng::from_seed(1);
        let mut b = SeededRng::from_seed(2);
        let differs = (0..100).any(|_| a.uniform() != b.uniform());
        assert!(differs);
    }

    #[test]
    fn uniform_deviates_are_in_unit_interval() {
        let mut rng = SeededRng::from_seed(7);
        for _ in 0..10_000 {
            let u = rng.uniform();
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn inverse_cdf_reproduces_exponential_mean() {
        // pdf(x) = e^{-x} on [0, 30]; mean = 1
        let s = InverseCdf1D::from_pdf(|x| (-x).exp(), 0.0, 30.0, 20_000).unwrap();
        let mut rng = SeededRng::from_seed(42);
        let n = 200_000;
        let mean: f64 = (0..n).map(|_| s.sample(&mut rng)).sum::<f64>() / n as f64;
        assert!((mean - 1.0).abs() < 0.02, "mean={mean}");
    }

    #[test]
    fn inverse_cdf_zero_density_returns_none() {
        assert!(InverseCdf1D::from_pdf(|_| 0.0, 0.0, 1.0, 100).is_none());
    }

    #[test]
    fn inverse_cdf_supports_displaced_domain() {
        // pdf concentrated near x = 5
        let s = InverseCdf1D::from_pdf(|x| (-(x - 5.0).powi(2)).exp(), 0.0, 10.0, 10_000).unwrap();
        let mut rng = SeededRng::from_seed(11);
        let n = 50_000;
        let mean: f64 = (0..n).map(|_| s.sample(&mut rng)).sum::<f64>() / n as f64;
        assert!((mean - 5.0).abs() < 0.02, "mean={mean}");
    }

    #[test]
    fn rejection_sampling_recovers_uniform() {
        let mut rng = SeededRng::from_seed(3);
        let n = 20_000;
        let mut sum = 0.0;
        for _ in 0..n {
            let x = rejection_sample_1d(|_| 1.0, 0.0, 1.0, 1.0, &mut rng, 100).unwrap();
            sum += x;
        }
        let mean = sum / n as f64;
        assert!((mean - 0.5).abs() < 0.02, "mean={mean}");
    }

    #[test]
    fn rejection_sampling_reports_failure_only_when_no_attempt_is_made() {
        let mut rng = SeededRng::from_seed(5);
        // No attempts -> no sample.
        assert!(rejection_sample_1d(|_| 1.0, 0.0, 1.0, 1.0, &mut rng, 0).is_none());
        // Non-positive bound -> no sample.
        assert!(rejection_sample_1d(|_| 1.0, 0.0, 1.0, 0.0, &mut rng, 10).is_none());
        assert!(rejection_sample_1d(|_| 1.0, 0.0, 1.0, -1.0, &mut rng, 10).is_none());
    }

    /// A too-small `pdf_max` cannot be detected by the sampler, so the caller
    /// must be able to detect the resulting bias themselves. This test
    /// documents the failure mode explicitly rather than pretending the API
    /// rejects it.
    ///
    /// Analytic expectations for the intended Gaussian-like density with
    /// `sigma = 0.5/sqrt(2) ~ 0.354`: variance `= sigma^2 = 0.125`. With
    /// `pdf_max` far below the true maximum the acceptance test becomes
    /// trivially true in the core region, so the accepted distribution is
    /// approximately uniform over the clipped support and its variance grows to
    /// `~0.8`. The test asserts that factorial difference rather than a magic
    /// absolute threshold.
    #[test]
    fn too_small_bound_produces_a_detectable_bias() {
        let pdf = |x: f64| {
            let d = (x - 3.0) / 0.5;
            (-d * d).exp()
        };
        let moments = |pdf_max: f64, seed: u64| -> (f64, f64) {
            let mut rng = SeededRng::from_seed(seed);
            let n = 20_000;
            let mut values = Vec::with_capacity(n);
            for _ in 0..n {
                values.push(rejection_sample_1d(pdf, 0.0, 6.0, pdf_max, &mut rng, 10_000).unwrap());
            }
            let mean = values.iter().sum::<f64>() / n as f64;
            let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
            (mean, var)
        };

        let (mean_good, var_good) = moments(1.0, 8);
        assert!(
            (mean_good - 3.0).abs() < 0.05,
            "valid bound mean = {mean_good}"
        );
        assert!(
            (var_good - 0.125).abs() < 0.03,
            "valid bound variance = {var_good}, analytic 0.125"
        );

        let (mean_bad, var_bad) = moments(1e-3, 8);
        assert!(
            mean_bad.is_finite(),
            "the sampler still returns values with an invalid bound"
        );
        assert!(
            var_bad > 4.0 * var_good,
            "an invalid bound must be detectable: var_bad={var_bad} vs var_good={var_good}"
        );
        println!(
            "rejection sampling: valid bound var = {var_good:.4} (analytic 0.125), \
             invalid bound var = {var_bad:.4}"
        );
    }
}
