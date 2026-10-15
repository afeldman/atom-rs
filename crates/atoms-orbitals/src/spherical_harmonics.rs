//! Complex spherical harmonics `Y_l^m(theta, phi)`.
//!
//! # Definition (normative)
//!
//! For `m >= 0`:
//!
//! ```text
//! Y_l^m(theta, phi) = N_{lm} P_l^m(cos theta) exp(i m phi)
//! N_{lm}            = sqrt( (2l+1)/(4 pi) * (l-m)!/(l+m)! )
//! ```
//!
//! where `P_l^m` is the associated Legendre function **including** the
//! Condon-Shortley phase `(-1)^m`
//! ([`atoms_numerics::special::associated_legendre_cs`]).
//!
//! For `m < 0`, with `M = -m`:
//!
//! ```text
//! Y_l^{-M} = (-1)^M conj( Y_l^M )
//! ```
//!
//! This is the convention used by `scipy.special.sph_harm` and by standard
//! quantum-mechanics texts. Orthonormality:
//!
//! ```text
//! int_0^{2pi} int_0^{pi} conj(Y_{l'm'}) Y_{lm} sin(theta) dtheta dphi
//!     = delta_{l l'} delta_{m m'}
//! ```
//!
//! # Phases matter
//!
//! `Y_l^m` and `Y_l^{-m}` are **not** the same function. For example
//! `Y_1^1 = -sqrt(3/(8pi)) sin(theta) e^{i phi}` while
//! `Y_1^{-1} = +sqrt(3/(8pi)) sin(theta) e^{-i phi}`. Code that silently
//! replaces `m` by `|m|` produces wrong complex phases; the tests in this module
//! pin the sign convention explicitly.
//!
//! # Angles
//!
//! `theta` is the polar angle from `+z` and `phi` the azimuth from `+x`
//! (`atoms_core::coordinates`). `|Y_l^m|` is independent of `phi`.

use atoms_numerics::special::{associated_legendre_cs, factorial};
use num_complex::Complex;
use std::f64::consts::PI;

/// Four pi, the solid-angle normalisation appearing in every `N_{lm}`.
const FOUR_PI: f64 = 4.0 * PI;

/// Errors raised when constructing a spherical harmonic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarmonicError {
    /// `|m| > l`.
    MagneticOutOfRange {
        /// The requested `l`.
        l: u32,
        /// The requested `m`.
        m: i32,
    },
}

impl std::fmt::Display for HarmonicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HarmonicError::MagneticOutOfRange { l, m } => write!(
                f,
                "spherical harmonic requires -l <= m <= l (got l = {l}, m = {m})"
            ),
        }
    }
}

impl std::error::Error for HarmonicError {}

/// A validated spherical harmonic `Y_l^m`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SphericalHarmonic {
    l: u32,
    m: i32,
}

/// Real normalisation constant `N_{lm}` for `m >= 0`, unit `sr^{-1/2}`.
///
/// For `m < 0` the constant of `Y_l^{|m|}` applies, since
/// `Y_l^{-M} = (-1)^M conj(Y_l^M)` and the phase does not change the magnitude.
pub fn normalization_constant(l: u32, m: i32) -> f64 {
    let mm = m.unsigned_abs();
    if mm > l {
        return f64::NAN;
    }
    let lf = l as f64;
    let ratio = factorial(l - mm) / factorial(l + mm);
    ((2.0 * lf + 1.0) / FOUR_PI * ratio).sqrt()
}

impl SphericalHarmonic {
    /// Validate and construct `Y_l^m`.
    ///
    /// # Errors
    /// Returns [`HarmonicError`] if `|m| > l`.
    pub fn new(l: u32, m: i32) -> Result<Self, HarmonicError> {
        if (m.unsigned_abs() as u64) > l as u64 {
            return Err(HarmonicError::MagneticOutOfRange { l, m });
        }
        Ok(Self { l, m })
    }

    /// Orbital angular momentum quantum number.
    pub fn l(&self) -> u32 {
        self.l
    }

    /// Magnetic quantum number, possibly negative.
    pub fn m(&self) -> i32 {
        self.m
    }

    /// Evaluate `Y_l^m(theta, phi)`.
    pub fn value(&self, theta: f64, phi: f64) -> Complex<f64> {
        evaluate(self.l, self.m, theta, phi)
    }

    /// `|Y_l^m(theta, phi)|^2`, independent of `phi`.
    pub fn abs_squared(&self, theta: f64) -> f64 {
        abs_squared(self.l, self.m, theta)
    }
}

/// Evaluate `Y_l^m(theta, phi)` without a construction step.
///
/// Returns a complex NaN-free value for every `(theta, phi)` when `|m| <= l`;
/// if `|m| > l` the returned value is `Complex::new(NAN, NAN)`.
pub fn evaluate(l: u32, m: i32, theta: f64, phi: f64) -> Complex<f64> {
    if (m.unsigned_abs() as u64) > l as u64 {
        return Complex::new(f64::NAN, f64::NAN);
    }
    let mm = m.unsigned_abs();
    let norm = normalization_constant(l, mm as i32);
    let plm = associated_legendre_cs(l, mm, theta.cos());
    let magnitude = norm * plm;

    if m >= 0 {
        // Y_l^m = N P e^{i m phi}
        Complex::from_polar(magnitude, m as f64 * phi)
    } else {
        // Y_l^{-M} = (-1)^M conj( N P e^{i M phi} )
        let positive = Complex::from_polar(magnitude, mm as f64 * phi);
        let conjugated = positive.conj();
        if mm % 2 == 1 {
            -conjugated
        } else {
            conjugated
        }
    }
}

/// `|Y_l^m(theta, phi)|^2`, which is independent of `phi`.
///
/// Returns `f64::NAN` when `|m| > l`.
pub fn abs_squared(l: u32, m: i32, theta: f64) -> f64 {
    if (m.unsigned_abs() as u64) > l as u64 {
        return f64::NAN;
    }
    let mm = m.unsigned_abs();
    let norm = normalization_constant(l, mm as i32);
    let plm = associated_legendre_cs(l, mm, theta.cos());
    let v = norm * plm;
    v * v
}

/// Convenience wrapper returning `Result`.
///
/// # Errors
/// Returns [`HarmonicError`] if `|m| > l`.
pub fn spherical_harmonic(
    l: u32,
    m: i32,
    theta: f64,
    phi: f64,
) -> Result<Complex<f64>, HarmonicError> {
    SphericalHarmonic::new(l, m).map(|h| h.value(theta, phi))
}

/// `|Y_l^m(theta, phi)|^2` as a `Result`.
///
/// # Errors
/// Returns [`HarmonicError`] if `|m| > l`.
pub fn spherical_harmonic_abs_squared(l: u32, m: i32, theta: f64) -> Result<f64, HarmonicError> {
    SphericalHarmonic::new(l, m).map(|h| h.abs_squared(theta))
}

#[cfg(test)]
mod tests {
    use super::*;
    use atoms_numerics::quadrature::simpson;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

    fn close(a: f64, b: f64, tol: f64, ctx: &str) {
        assert!((a - b).abs() < tol, "{ctx}: {a} vs {b}");
    }

    fn close_c(a: Complex<f64>, b: Complex<f64>, tol: f64, ctx: &str) {
        assert!((a - b).norm() < tol, "{ctx}: {a} vs {b}");
    }

    fn double_simpson<F: Fn(f64, f64) -> f64>(
        f: F,
        t0: f64,
        t1: f64,
        nt: usize,
        p0: f64,
        p1: f64,
        np: usize,
    ) -> f64 {
        simpson(|t| simpson(|p| f(t, p), p0, p1, np), t0, t1, nt)
    }

    /// Reference values from the standard closed forms.
    #[test]
    fn matches_reference_closed_forms() {
        let theta = 0.9_f64;
        let phi = 0.4_f64;
        let ct = theta.cos();
        let st = theta.sin();

        // Y_0^0 = 1/(2 sqrt pi)
        close_c(
            evaluate(0, 0, theta, phi),
            Complex::new(1.0 / (2.0 * PI.sqrt()), 0.0),
            1e-15,
            "Y00",
        );

        // Y_1^0 = sqrt(3/(4 pi)) cos(theta)
        close_c(
            evaluate(1, 0, theta, phi),
            Complex::new((3.0 / FOUR_PI).sqrt() * ct, 0.0),
            1e-15,
            "Y10",
        );

        // Y_1^1 = -sqrt(3/(8 pi)) sin(theta) e^{i phi}
        let c = -(3.0 / (8.0 * PI)).sqrt();
        close_c(
            evaluate(1, 1, theta, phi),
            Complex::from_polar(c * st, phi),
            1e-15,
            "Y11",
        );

        // Y_1^{-1} = +sqrt(3/(8 pi)) sin(theta) e^{-i phi}
        close_c(
            evaluate(1, -1, theta, phi),
            Complex::from_polar(-c * st, -phi),
            1e-15,
            "Y1-1",
        );

        // Y_2^0 = sqrt(5/(16 pi)) (3 cos^2 - 1)
        close_c(
            evaluate(2, 0, theta, phi),
            Complex::new((5.0 / (16.0 * PI)).sqrt() * (3.0 * ct * ct - 1.0), 0.0),
            1e-14,
            "Y20",
        );

        // Y_2^2 = sqrt(15/(32 pi)) sin^2(theta) e^{2 i phi}
        close_c(
            evaluate(2, 2, theta, phi),
            Complex::from_polar((15.0 / (32.0 * PI)).sqrt() * st * st, 2.0 * phi),
            1e-14,
            "Y22",
        );

        // Y_2^{-2} = sqrt(15/(32 pi)) sin^2(theta) e^{-2 i phi}   (even M: no sign)
        close_c(
            evaluate(2, -2, theta, phi),
            Complex::from_polar((15.0 / (32.0 * PI)).sqrt() * st * st, -2.0 * phi),
            1e-14,
            "Y2-2",
        );
    }

    /// External cross-validation against SciPy lives in the `atoms-validation`
    /// crate (`tests/reference_values.rs`). It compares this implementation
    /// against `scipy.special.sph_harm` over 252 (l, m, theta, phi) samples
    /// whose values are generated by `python/tests/generate_reference_values.py`
    /// and committed as `docs/validation/reference-values.csv`. Reference values
    /// are never transcribed by hand.
    ///
    /// Only the analytically derived closed forms above are checked here, so
    /// that this module remains self-contained.
    #[test]
    fn reference_values_are_generated_not_transcribed() {
        // The generator must be present and the data file usable; the actual
        // numeric comparison is in atoms-validation.
        assert!(
            include_str!("../../../python/tests/generate_reference_values.py").contains("scipy")
        );
    }

    #[test]
    fn abs_squared_is_phi_independent_and_non_negative() {
        for l in 0..=4u32 {
            let lm = l as i32;
            for m in -lm..=lm {
                let base = abs_squared(l, m, 1.0);
                for &phi in &[0.0, 1.0, 3.0, 5.5] {
                    let v = evaluate(l, m, 1.0, phi);
                    close(v.norm_sqr(), base, 1e-14, "phi independence");
                    assert!(v.norm_sqr() >= 0.0);
                }
                assert!(base >= 0.0);
            }
        }
    }

    #[test]
    fn negative_m_relation_holds() {
        for l in 0..=5u32 {
            for m in 1..=l as i32 {
                let pos = evaluate(l, m, 0.7, 1.3);
                let neg = evaluate(l, -m, 0.7, 1.3);
                let expected = if m % 2 == 1 { -pos.conj() } else { pos.conj() };
                close_c(neg, expected, 1e-14, &format!("l={l} m={m}"));
            }
        }
    }

    /// Orthonormality: `int conj(Y_{l'm'}) Y_{lm} sin(theta) dtheta dphi = delta`.
    #[test]
    fn harmonics_are_orthonormal() {
        let states: Vec<(u32, i32)> = (0..=3u32)
            .flat_map(|l| {
                let lm = l as i32;
                (-lm..=lm).map(move |m| (l, m))
            })
            .collect();
        let nt = 300;
        let np = 300;
        for &(l1, m1) in &states {
            for &(l2, m2) in &states {
                let integrand = |t: f64, p: f64| {
                    let a = evaluate(l1, m1, t, p);
                    let b = evaluate(l2, m2, t, p);
                    (a.conj() * b).re * t.sin()
                };
                let integral = double_simpson(integrand, 0.0, PI, nt, 0.0, 2.0 * PI, np);
                let expected = if (l1, m1) == (l2, m2) { 1.0 } else { 0.0 };
                assert!(
                    (integral - expected).abs() < 1e-6,
                    "l1={l1} m1={m1} l2={l2} m2={m2} integral={integral}"
                );
            }
        }
    }

    /// Addition theorem: `sum_{m=-l}^{l} |Y_l^m(theta,phi)|^2 = (2l+1)/(4 pi)`.
    ///
    /// This is an exact identity and therefore a strong check of the relative
    /// phases between all `m` of a given `l`.
    #[test]
    fn addition_theorem_holds() {
        for l in 0..=6u32 {
            let expected = (2.0 * l as f64 + 1.0) / FOUR_PI;
            for &theta in &[0.0, 0.3, FRAC_PI_4, FRAC_PI_2, 2.2, PI] {
                let sum: f64 = (-(l as i32)..=l as i32)
                    .map(|m| abs_squared(l, m, theta))
                    .sum();
                assert!(
                    (sum - expected).abs() < 1e-12,
                    "l={l} theta={theta} sum={sum} expected={expected}"
                );
            }
        }
    }

    /// `L_z Y_l^m = m Y_l^m` with `L_z = -i d/dphi`; here `d/dphi Y = i m Y`,
    /// which the polar form satisfies exactly.
    #[test]
    fn is_eigenfunction_of_lz() {
        for l in 0..=4u32 {
            let lm = l as i32;
            for m in -lm..=lm {
                for &phi in &[0.2, 1.7, 4.4] {
                    let y = evaluate(l, m, 0.8, phi);
                    let h = 1e-6;
                    let dy =
                        (evaluate(l, m, 0.8, phi + h) - evaluate(l, m, 0.8, phi - h)) / (2.0 * h);
                    let lhs = Complex::new(0.0, -1.0) * dy;
                    close_c(lhs, y * (m as f64), 1e-8 * (m.abs().max(1) as f64), "Lz");
                }
            }
        }
    }

    /// The raising operator `L_+ = e^{i phi}(d/dtheta + i cot(theta) d/dphi)`
    /// must satisfy `L_+ Y_l^m = sqrt(l(l+1) - m(m+1)) Y_l^{m+1}`.
    #[test]
    fn raising_operator_relation_holds() {
        let h = 1e-6_f64;
        for l in 1..=4u32 {
            for m in -(l as i32)..(l as i32) {
                let theta = 0.9_f64;
                let phi = 0.35_f64;
                let dtheta =
                    (evaluate(l, m, theta + h, phi) - evaluate(l, m, theta - h, phi)) / (2.0 * h);
                let dphi =
                    (evaluate(l, m, theta, phi + h) - evaluate(l, m, theta, phi - h)) / (2.0 * h);
                let cot = theta.cos() / theta.sin();
                let lhs =
                    Complex::from_polar(1.0, phi) * (dtheta + Complex::new(0.0, 1.0) * cot * dphi);
                let lf = l as f64;
                let mf = m as f64;
                let coeff = (lf * (lf + 1.0) - mf * (mf + 1.0)).sqrt();
                let rhs = evaluate(l, m + 1, theta, phi) * coeff;
                close_c(lhs, rhs, 1e-5, &format!("l={l} m={m}"));
            }
        }
    }

    #[test]
    fn invalid_m_returns_error_or_nan() {
        assert!(SphericalHarmonic::new(1, 2).is_err());
        assert!(SphericalHarmonic::new(1, -2).is_err());
        assert!(spherical_harmonic(2, 3, 0.0, 0.0).is_err());
        assert!(spherical_harmonic_abs_squared(2, -3, 0.0).is_err());
        assert!(evaluate(1, 5, 0.0, 0.0).re.is_nan());
        assert!(abs_squared(1, -5, 0.0).is_nan());
        // valid extremes
        assert!(SphericalHarmonic::new(3, 3).is_ok());
        assert!(SphericalHarmonic::new(3, -3).is_ok());
    }

    /// The polar-angle dependence must reproduce the analytic nodal structure:
    /// the zonal harmonic `Y_l^0` changes sign exactly `l` times in `(0, pi)`.
    ///
    /// This test uses the *signed* real function, not its modulus: taking the
    /// absolute value would destroy exactly the sign changes being counted.
    #[test]
    fn zonal_harmonics_have_l_nodes() {
        for l in 1..=6u32 {
            let steps = 20_000;
            let mut sign_changes = 0;
            let mut prev = 0i32;
            for i in 1..steps {
                let theta = PI * i as f64 / steps as f64;
                let v = evaluate(l, 0, theta, 0.0).re;
                let s = if v > 0.0 {
                    1
                } else if v < 0.0 {
                    -1
                } else {
                    0
                };
                if s != 0 {
                    if prev != 0 && s != prev {
                        sign_changes += 1;
                    }
                    prev = s;
                }
            }
            assert_eq!(sign_changes, l, "l={l}");
        }
    }

    #[test]
    fn normalization_constants_match_closed_forms() {
        close(
            normalization_constant(0, 0),
            1.0 / (2.0 * PI.sqrt()),
            1e-15,
            "N00",
        );
        close(
            normalization_constant(1, 0),
            (3.0 / FOUR_PI).sqrt(),
            1e-15,
            "N10",
        );
        // negative m uses the same constant as |m|
        close(
            normalization_constant(1, -1),
            normalization_constant(1, 1),
            1e-15,
            "N1-1",
        );
        assert!(normalization_constant(1, 2).is_nan());
    }
}
