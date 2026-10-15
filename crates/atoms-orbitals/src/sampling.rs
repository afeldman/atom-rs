//! Deterministic sampling of hydrogenic probability densities.
//!
//! Sampling is a *scientific* operation, not a presentation detail, so it lives
//! in the physics crate rather than in the language bindings. The Python
//! bindings call [`OrbitalSampler`] and only handle array marshalling.
//!
//! # Method
//!
//! `|psi_{nlm}|^2 = R_{nl}(r)^2 |Y_l^m(theta)|^2` factorises, and `|Y_l^m|` is
//! independent of `phi`. A sample is therefore drawn in three independent steps:
//!
//! 1. `r` from the radial density `P(r) = r^2 R_{nl}(r)^2` on `[0, r_max]`, by
//!    inverse-CDF sampling;
//! 2. `mu = cos(theta)` from `|P_l^{|m|}(mu)|^2` on `[-1, 1]`, by inverse-CDF
//!    sampling. Working in `mu` rather than `theta` removes the `sin(theta)`
//!    Jacobian from the density and makes the tabulation uniform in the sampled
//!    variable;
//! 3. `phi` uniformly on `[0, 2 pi)`.
//!
//! # Truncation
//!
//! `r_max` is set to `8 <r> + 30` bohr, with `<r>` the analytic expectation
//! value. The neglected tail is bounded and measured in the test suite
//! (`P(r > r_max) < 1e-20` for the states tested); the tabulated density is
//! renormalised on `[0, r_max]`, so the returned samples are drawn from the
//! truncated distribution, whose total mass differs from 1 by less than that
//! bound.
//!
//! # Determinism
//!
//! Every sample is a function of the caller's [`SeededRng`] state alone. The
//! same seed reproduces the same sequence on every platform.

use crate::radial::radial_unchecked;
use crate::wavefunction::HydrogenWavefunction;
use atoms_numerics::sampling::{InverseCdf1D, SeededRng};
use atoms_numerics::special::associated_legendre_cs;
use std::f64::consts::TAU;

/// Upper bound on the sampled radial domain, in bohr, for a state with analytic
/// expectation value `mean_r`.
pub fn sampling_radius_max(mean_r: f64) -> f64 {
    8.0 * mean_r + 30.0
}

/// Reasons an [`OrbitalSampler`] cannot be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamplerError {
    /// The tabulated radial density integrated to zero on the sampling domain.
    EmptyRadialSupport,
    /// The tabulated angular density integrated to zero on `[-1, 1]`.
    EmptyAngularSupport,
}

impl std::fmt::Display for SamplerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SamplerError::EmptyRadialSupport => {
                write!(
                    f,
                    "the radial probability density is zero on the sampling domain"
                )
            }
            SamplerError::EmptyAngularSupport => {
                write!(f, "the angular probability density is zero on [-1, 1]")
            }
        }
    }
}

impl std::error::Error for SamplerError {}

/// A sampler for one orbital.
#[derive(Debug, Clone)]
pub struct OrbitalSampler {
    radial: InverseCdf1D,
    angular: InverseCdf1D,
    radial_max: f64,
    radial_table_points: usize,
    angular_table_points: usize,
}

impl OrbitalSampler {
    /// Build samplers for `wf`.
    ///
    /// Table sizes scale with the number of radial nodes so that the nodes are
    /// resolved even for high `n`.
    ///
    /// # Errors
    /// Returns [`SamplerError`] if a tabulated density integrates to zero, which
    /// cannot happen for a valid hydrogenic state and therefore indicates a bug.
    pub fn new(wf: &HydrogenWavefunction) -> Result<Self, SamplerError> {
        let qn = wf.quantum_numbers();
        let n = qn.n();
        let l = qn.l();
        let m = qn.m();

        let mean_r = crate::expectations::expectation_r(n, l) / wf.z() as f64;
        let radial_max = sampling_radius_max(mean_r);

        // Radially: more nodes need more table points.
        let radial_table_points = 20_000 + 2_000 * n as usize;

        let radial = {
            let z = wf.z() as f64;
            let rmax = radial_max;
            let pdf = move |r: f64| {
                if r < 0.0 || r > rmax {
                    return 0.0;
                }
                let rad = z.powf(1.5) * radial_unchecked(n, l, z * r);
                r * r * rad * rad
            };
            InverseCdf1D::from_pdf(pdf, 0.0, radial_max, radial_table_points)
                .ok_or(SamplerError::EmptyRadialSupport)?
        };

        // Angularly: |Y_l^m|^2 = N^2 P_l^{|m|}(mu)^2, independent of phi.
        let mm = m.unsigned_abs();
        // Use the same normalisation path as the public API so that a phase or
        // normalisation regression in the harmonics is caught here too.
        let norm = crate::spherical_harmonics::normalization_constant(l, m);
        let angular = {
            let pdf = move |mu: f64| {
                let p = associated_legendre_cs(l, mm, mu);
                let v = norm * p;
                v * v
            };
            InverseCdf1D::from_pdf(pdf, -1.0, 1.0, 20_000)
                .ok_or(SamplerError::EmptyAngularSupport)?
        };

        Ok(Self {
            radial,
            angular,
            radial_max,
            radial_table_points,
            angular_table_points: 20_000,
        })
    }

    /// Upper bound of the sampled radial domain, in bohr.
    pub fn radial_max(&self) -> f64 {
        self.radial_max
    }

    /// Number of table points used for the radial density.
    pub fn radial_table_points(&self) -> usize {
        self.radial_table_points
    }

    /// Number of table points used for the angular density.
    pub fn angular_table_points(&self) -> usize {
        self.angular_table_points
    }

    /// Draw `r` in bohr from `P(r)`.
    pub fn sample_radius(&self, rng: &mut SeededRng) -> f64 {
        self.radial.sample(rng)
    }

    /// Draw `mu = cos(theta)` in `[-1, 1]`.
    pub fn sample_mu(&self, rng: &mut SeededRng) -> f64 {
        self.angular.sample(rng)
    }

    /// Draw `(theta, phi)` in radians, with `theta` the polar angle from `+z`.
    pub fn sample_direction(&self, rng: &mut SeededRng) -> (f64, f64) {
        // Draw both so that the RNG consumption per sample is constant; this
        // keeps a given seed reproducible independently of which accessors a
        // caller uses in sequence.
        let mu = self.sample_mu(rng);
        let phi = rng.uniform_range(0.0, TAU);
        (mu.clamp(-1.0, 1.0).acos(), phi)
    }

    /// Draw a Cartesian position in bohr from `|psi|^2`.
    pub fn sample_cartesian(&self, rng: &mut SeededRng) -> [f64; 3] {
        let r = self.sample_radius(rng);
        let (theta, phi) = self.sample_direction(rng);
        let st = theta.sin();
        [r * st * phi.cos(), r * st * phi.sin(), r * theta.cos()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HydrogenWavefunction;

    fn mean_of(samples: &[f64]) -> f64 {
        samples.iter().sum::<f64>() / samples.len() as f64
    }

    #[test]
    fn one_s_radius_reproduces_analytic_mean_and_variance() {
        let wf = HydrogenWavefunction::new(1, 0, 0).unwrap();
        let s = OrbitalSampler::new(&wf).unwrap();
        let mut rng = SeededRng::from_seed(20261015);
        let n = 400_000;
        let mut rs = Vec::with_capacity(n);
        let mut sq = 0.0;
        for _ in 0..n {
            let r = s.sample_radius(&mut rng);
            rs.push(r);
            sq += r * r;
        }
        let mean = mean_of(&rs);
        let var = sq / n as f64 - mean * mean;
        // standard error sqrt(0.75/400000) ~ 0.0014
        assert!((mean - 1.5).abs() < 0.01, "mean = {mean}");
        assert!((var - 0.75).abs() < 0.02, "var = {var}");
    }

    #[test]
    fn two_p_radius_reproduces_analytic_mean() {
        let wf = HydrogenWavefunction::new(2, 1, 0).unwrap();
        let s = OrbitalSampler::new(&wf).unwrap();
        let mut rng = SeededRng::from_seed(7);
        let n = 200_000;
        let mean = mean_of(
            &(0..n)
                .map(|_| s.sample_radius(&mut rng))
                .collect::<Vec<_>>(),
        );
        assert!((mean - 5.0).abs() < 0.03, "mean = {mean}");
    }

    /// `mu` must follow `|P_l^m|^2`: for `l = 0` that is uniform, giving
    /// `<cos^2 theta> = 1/3`; for `l = 1, m = 0` it gives `3/5`; for `l = 1,
    /// |m| = 1` it gives `1/5`.
    #[test]
    fn angular_sampling_matches_the_legendre_density() {
        let cases: [(u32, i32, f64); 5] = [
            (0, 0, 1.0 / 3.0),
            (1, 0, 3.0 / 5.0),
            (1, 1, 1.0 / 5.0),
            (2, 0, 11.0 / 21.0),
            (2, 2, 1.0 / 7.0),
        ];
        for (l, m, expected) in cases {
            let wf = HydrogenWavefunction::new(l + 1, l, m).unwrap();
            let s = OrbitalSampler::new(&wf).unwrap();
            let mut rng = SeededRng::from_seed(99);
            let n = 200_000;
            let mut acc = 0.0;
            for _ in 0..n {
                let mu = s.sample_mu(&mut rng);
                acc += mu * mu;
            }
            let got = acc / n as f64;
            assert!(
                (got - expected).abs() < 0.01,
                "l={l} m={m}: <cos^2 theta> = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn direction_is_in_range_and_phi_is_uniform() {
        let wf = HydrogenWavefunction::new(2, 1, -1).unwrap();
        let s = OrbitalSampler::new(&wf).unwrap();
        let mut rng = SeededRng::from_seed(31);
        let n = 50_000;
        let mut sx = 0.0;
        let mut sy = 0.0;
        for _ in 0..n {
            let (theta, phi) = s.sample_direction(&mut rng);
            assert!(
                (0.0..=std::f64::consts::PI).contains(&theta),
                "theta={theta}"
            );
            assert!((0.0..TAU).contains(&phi), "phi={phi}");
            sx += phi.cos();
            sy += phi.sin();
        }
        // uniform phi -> both circular means vanish
        assert!((sx / n as f64).abs() < 0.02);
        assert!((sy / n as f64).abs() < 0.02);
    }

    #[test]
    fn cartesian_samples_have_the_expected_radial_moments() {
        let wf = HydrogenWavefunction::new(3, 2, 0).unwrap();
        let s = OrbitalSampler::new(&wf).unwrap();
        let mut rng = SeededRng::from_seed(5);
        let n = 100_000;
        let mut acc = 0.0;
        for _ in 0..n {
            let [x, y, z] = s.sample_cartesian(&mut rng);
            acc += (x * x + y * y + z * z).sqrt();
        }
        let mean = acc / n as f64;
        let expected = crate::expectations::expectation_r(3, 2);
        assert!(
            (mean - expected).abs() < 0.05,
            "3d <r> = {mean}, expected {expected}"
        );
    }

    #[test]
    fn sampling_is_deterministic_for_a_seed() {
        let wf = HydrogenWavefunction::new(3, 2, -1).unwrap();
        let s = OrbitalSampler::new(&wf).unwrap();
        let draw = |seed: u64| {
            let mut rng = SeededRng::from_seed(seed);
            (0..2000)
                .map(|_| s.sample_cartesian(&mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(1234), draw(1234));
        assert_ne!(draw(1234), draw(1235));
    }

    /// The tabulated domain must contain essentially all of the probability.
    #[test]
    fn radial_domain_contains_the_probability() {
        use atoms_numerics::quadrature::simpson;
        for (n, l) in [(1u32, 0u32), (3, 2), (5, 0), (8, 3)] {
            let wf = HydrogenWavefunction::new(n, l, 0).unwrap();
            let s = OrbitalSampler::new(&wf).unwrap();
            let tail = simpson(
                |r| wf.radial_probability_density(r),
                s.radial_max(),
                s.radial_max() + 200.0,
                200_000,
            );
            assert!(tail < 1e-20, "n={n} l={l}: tail = {tail:e}");
        }
    }

    #[test]
    fn hydrogenic_ion_sampling_is_contracted() {
        // He+ 1s: <r> = 1.5/Z = 0.75
        let wf = HydrogenWavefunction::hydrogenic(1, 0, 0, 2).unwrap();
        let s = OrbitalSampler::new(&wf).unwrap();
        let mut rng = SeededRng::from_seed(11);
        let n = 200_000;
        let mean = mean_of(
            &(0..n)
                .map(|_| s.sample_radius(&mut rng))
                .collect::<Vec<_>>(),
        );
        assert!((mean - 0.75).abs() < 0.01, "mean = {mean}");
    }

    #[test]
    fn table_sizes_scale_with_the_principal_quantum_number() {
        let low = OrbitalSampler::new(&HydrogenWavefunction::new(1, 0, 0).unwrap()).unwrap();
        let high = OrbitalSampler::new(&HydrogenWavefunction::new(20, 0, 0).unwrap()).unwrap();
        assert!(high.radial_table_points() > low.radial_table_points());
        assert!(high.radial_max() > low.radial_max());
    }
}
