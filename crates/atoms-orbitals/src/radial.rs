//! Normalised hydrogen-like radial wavefunctions `R_{nl}(r)`.
//!
//! # Definition
//!
//! In Hartree atomic units, with `rho = 2 r / n`,
//!
//! ```text
//! R_{nl}(r) = N_{nl} exp(-rho/2) rho^l L_{n-l-1}^{2l+1}(rho)
//! N_{nl}    = sqrt( (2/n)^3 (n-l-1)! / (2 n (n+l)!) )
//! ```
//!
//! `L` is the generalised Laguerre polynomial as implemented in
//! [`atoms_numerics::special::generalized_laguerre`]. The normalisation is
//!
//! ```text
//! int_0^inf r^2 R_{nl}(r)^2 dr = 1
//! ```
//!
//! # Nuclear charge
//!
//! For a one-electron ion of nuclear charge `Z` the radial function scales as
//! `R_{nl}^{Z}(r) = Z^{3/2} R_{nl}(Zr)`, which preserves the normalisation and
//! shrinks the orbital by `1/Z`. `Z > 1` is the **unscreened** hydrogenic
//! approximation; it is not a multi-electron description.
//!
//! # Supported range
//!
//! Validated for `1 <= n <= 40`. Beyond that the Laguerre recurrence loses
//! accuracy in the deeply classical region and no claim is made.
//!
//! # Units
//!
//! `r` is in bohr. `R` has unit `bohr^{-3/2}`; `r^2 R^2` has unit `bohr^{-1}`.

use crate::quantum_numbers::{QuantumNumberError, QuantumNumbers};
use atoms_numerics::special::{factorial, generalized_laguerre};

/// Largest principal quantum number for which this module makes a validated
/// accuracy claim.
pub const MAX_VALIDATED_N: u32 = 40;

/// Normalisation constant `N_{nl}` (unit `bohr^{-3/2}`).
///
/// Preconditions `n >= 1` and `l < n` are the caller's responsibility; use
/// [`radial_wavefunction`] for the checked entry point.
pub fn normalization_factor(n: u32, l: u32) -> f64 {
    let nf = n as f64;
    let prefactor = (2.0 / nf).powi(3);
    let ratio = factorial(n - l - 1) / (2.0 * nf * factorial(n + l));
    (prefactor * ratio).sqrt()
}

/// `R_{nl}(r)` without quantum-number validation.
///
/// Returns `f64::NAN` for `r < 0`. Preconditions `n >= 1`, `l < n`.
pub(crate) fn radial_unchecked(n: u32, l: u32, r: f64) -> f64 {
    if r < 0.0 {
        return f64::NAN;
    }
    let nf = n as f64;
    let rho = 2.0 * r / nf;
    let norm = normalization_factor(n, l);
    let lag = generalized_laguerre(n - l - 1, (2 * l + 1) as f64, rho);
    norm * (-rho / 2.0).exp() * rho.powi(l as i32) * lag
}

/// Normalised radial wavefunction `R_{nl}(r)` in bohr^{-3/2}.
///
/// # Errors
/// Returns [`QuantumNumberError`] if `n == 0` or `l >= n`.
pub fn radial_wavefunction(n: u32, l: u32, r: f64) -> Result<f64, QuantumNumberError> {
    QuantumNumbers::new(n, l, 0)?;
    Ok(radial_unchecked(n, l, r))
}

/// Normalised radial wavefunction for a one-electron ion of nuclear charge `Z`.
///
/// Uses `R_{nl}^{Z}(r) = Z^{3/2} R_{nl}(Zr)`. See the module documentation for
/// the meaning and limitations of `Z > 1`.
///
/// # Errors
/// Returns [`QuantumNumberError`] if `n == 0` or `l >= n`.
pub fn radial_wavefunction_z(n: u32, l: u32, z: u32, r: f64) -> Result<f64, QuantumNumberError> {
    QuantumNumbers::new(n, l, 0)?;
    if z == 0 {
        return Ok(f64::NAN);
    }
    let zf = z as f64;
    Ok(zf.powf(1.5) * radial_unchecked(n, l, zf * r))
}

/// Radial probability density `P(r) = r^2 |R_{nl}(r)|^2` in bohr^{-1}.
///
/// `P(r) dr` is the probability of finding the electron with radial coordinate
/// in `[r, r + dr)`, i.e. it is the density of the radial measure obtained after
/// integrating over all angles, and it satisfies
/// `int_0^inf P(r) dr = 1`.
///
/// This is **not** the same quantity as `|psi|^2`: see the crate documentation.
/// In particular `P(0) = 0` for every orbital, whereas `|psi(0)|^2 != 0` for
/// s orbitals.
///
/// # Errors
/// Returns [`QuantumNumberError`] if `n == 0` or `l >= n`.
pub fn radial_probability_density(n: u32, l: u32, r: f64) -> Result<f64, QuantumNumberError> {
    let rr = radial_wavefunction(n, l, r)?;
    Ok(r * r * rr * rr)
}

/// Radial probability density for a one-electron ion of nuclear charge `Z`,
/// in bohr^{-1}.
///
/// # Errors
/// Returns [`QuantumNumberError`] if `n == 0` or `l >= n`.
pub fn radial_probability_density_z(
    n: u32,
    l: u32,
    z: u32,
    r: f64,
) -> Result<f64, QuantumNumberError> {
    let rr = radial_wavefunction_z(n, l, z, r)?;
    Ok(r * r * rr * rr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use atoms_numerics::quadrature::simpson;
    use std::f64::consts::SQRT_2;

    const TOL: f64 = 1e-12;

    fn rel(a: f64, b: f64) -> f64 {
        if b == 0.0 {
            a.abs()
        } else {
            ((a - b) / b).abs()
        }
    }

    /// Closed forms from the standard textbook normalisation.
    #[test]
    fn radial_functions_match_closed_forms() {
        let samples: [f64; 6] = [0.0, 0.25, 1.0, 2.5, 5.0, 12.0];

        for &r in &samples {
            // R_10 = 2 e^{-r}
            let want = 2.0 * (-r).exp();
            assert!(
                rel(radial_wavefunction(1, 0, r).unwrap(), want) < TOL,
                "1s r={r}"
            );

            // R_20 = (1/(2 sqrt 2)) (2 - r) e^{-r/2}
            let want = (1.0 / (2.0 * SQRT_2)) * (2.0 - r) * (-r / 2.0).exp();
            assert!(
                rel(radial_wavefunction(2, 0, r).unwrap(), want) < TOL,
                "2s r={r}"
            );

            // R_21 = (1/(2 sqrt 6)) r e^{-r/2}
            let want = r * (-r / 2.0).exp() / (2.0 * 6.0_f64.sqrt());
            assert!(
                rel(radial_wavefunction(2, 1, r).unwrap(), want) < TOL,
                "2p r={r}"
            );

            // R_30 = (2/(81 sqrt 3)) (27 - 18 r + 2 r^2) e^{-r/3}
            let want = (2.0 / (81.0 * 3.0_f64.sqrt()))
                * (27.0 - 18.0 * r + 2.0 * r * r)
                * (-r / 3.0).exp();
            assert!(
                rel(radial_wavefunction(3, 0, r).unwrap(), want) < 1e-11,
                "3s r={r}"
            );

            // R_32 = (4/(81 sqrt 30)) r^2 e^{-r/3}
            let want = (4.0 / (81.0 * 30.0_f64.sqrt())) * r * r * (-r / 3.0).exp();
            assert!(
                rel(radial_wavefunction(3, 2, r).unwrap(), want) < 1e-11,
                "3d r={r}"
            );
        }
    }

    #[test]
    fn normalization_integral_is_one() {
        // int_0^inf r^2 R^2 dr = 1, computed by truncation well beyond the
        // classically allowed region and Simpson's rule.
        for n in 1..=6u32 {
            for l in 0..n {
                let r_max = 6.0 * (n * n) as f64 + 20.0;
                let f = |r: f64| {
                    let v = radial_unchecked(n, l, r);
                    r * r * v * v
                };
                let total = simpson(f, 0.0, r_max, 200_000);
                assert!((total - 1.0).abs() < 1e-8, "n={n} l={l} integral={total}");
            }
        }
    }

    /// `P(r) = r^2 R^2` integrates to 1 as well, and is zero at the origin for
    /// every orbital, including the s orbitals.
    #[test]
    fn radial_probability_density_normalised_and_zero_at_origin() {
        for (n, l) in [(1u32, 0u32), (2, 0), (2, 1), (3, 2), (4, 3)] {
            assert_eq!(radial_probability_density(n, l, 0.0).unwrap(), 0.0);
            let r_max = 6.0 * (n * n) as f64 + 30.0;
            let total = simpson(
                |r| r * r * radial_unchecked(n, l, r).powi(2),
                0.0,
                r_max,
                200_000,
            );
            assert!((total - 1.0).abs() < 1e-8, "n={n} l={l}");
        }
    }

    /// The near-origin scaling for `l >= 1` must be exactly `r^l`: the
    /// exponential factor `e^{-r/n}` tends to 1 as `r -> 0`, so the ratio of `R`
    /// at `2r` and `r` must approach `2^l`.
    #[test]
    fn origin_behaviour_follows_r_to_the_l() {
        assert!(rel(radial_wavefunction(1, 0, 0.0).unwrap(), 2.0) < 1e-14);
        for l in 1..4u32 {
            assert_eq!(radial_wavefunction(4, l, 0.0).unwrap(), 0.0, "l={l}");
        }
        // Small enough r that e^{-r/n} differs from 1 by less than 1e-8.
        for l in 1..4u32 {
            let r = 1e-8_f64;
            let a = radial_wavefunction(4, l, r).unwrap();
            let b = radial_wavefunction(4, l, 2.0 * r).unwrap();
            let expected = (2.0_f64).powi(l as i32);
            assert!(
                (b / a - expected).abs() < 1e-6 * expected,
                "l={l} b/a={} expected={expected}",
                b / a
            );
        }
        // At a finite radius the exponential factor must be visible:
        // R ~ r e^{-r/2}, so b/a = 2 e^{-dr/2} with dr = 1e-4.
        let a = radial_wavefunction(2, 1, 1e-4).unwrap();
        let b = radial_wavefunction(2, 1, 2e-4).unwrap();
        let expected = 2.0 * (-5e-5_f64).exp();
        assert!((b / a - expected).abs() < 1e-12, "b/a={}", b / a);
    }

    /// `E_n = -Z^2/(2n^2)` for the pure Coulomb problem; the radial function
    /// with `Z` must reproduce the `1/Z` length contraction exactly.
    #[test]
    fn nuclear_charge_scaling_is_exact() {
        for (n, l) in [(1u32, 0u32), (2, 1), (3, 2)] {
            for &r in &[0.3, 1.0, 4.0] {
                let z = 2u32;
                let a = radial_wavefunction_z(n, l, z, r).unwrap();
                let b = (z as f64).powf(1.5) * radial_wavefunction(n, l, z as f64 * r).unwrap();
                assert!(rel(a, b) < 1e-13, "n={n} l={l} r={r}");
            }
        }
    }

    #[test]
    fn nuclear_charge_preserves_normalization() {
        for z in [1u32, 2, 3, 6] {
            let total = simpson(
                |r| {
                    let v = radial_wavefunction_z(1, 0, z, r).unwrap();
                    r * r * v * v
                },
                0.0,
                4.0 / z as f64 + 20.0,
                200_000,
            );
            assert!((total - 1.0).abs() < 1e-8, "z={z} total={total}");
        }
    }

    #[test]
    fn rejects_invalid_quantum_numbers() {
        assert!(radial_wavefunction(0, 0, 1.0).is_err());
        assert!(radial_wavefunction(1, 1, 1.0).is_err());
        assert!(radial_wavefunction(2, 2, 1.0).is_err());
        assert!(radial_probability_density(0, 0, 1.0).is_err());
    }

    #[test]
    fn negative_radius_is_nan_not_a_silent_number() {
        assert!(radial_wavefunction(1, 0, -1.0).unwrap().is_nan());
    }

    #[test]
    fn large_radius_decays_exponentially_with_scale_n() {
        // R ~ exp(-r/n): doubling r at large r must shrink R by exp(-r/n)
        for n in 1..=4u32 {
            let r = 8.0 * n as f64;
            let a = radial_unchecked(n, 0, r);
            let b = radial_unchecked(n, 0, r + n as f64);
            let ratio = b / a;
            assert!(ratio < 1.0);
            // loosely between exp(-1) and exp(-0.5) given polynomial prefactors
            assert!(ratio < 0.75 && ratio > 0.1, "n={n} ratio={ratio}");
        }
    }

    /// An independent check of the implementation against the **defining**
    /// differential equation rather than against another evaluation of the same
    /// closed form.
    ///
    /// `R` must satisfy the radial Schrodinger equation
    /// `-1/2 R'' - R'/r + [l(l+1)/(2r^2) - 1/r] R = E R` with `E = -1/(2 n^2)`.
    /// The residual is measured against the maximum amplitude of `R` for that
    /// `(n, l)` so that points close to a radial node — where `R` itself is
    /// vanishingly small and the *relative* residual is meaningless — do not
    /// produce spurious failures.
    #[test]
    fn satisfies_radial_schrodinger_equation() {
        let h = 1e-4_f64;
        for (n, l) in [(1u32, 0u32), (2, 0), (2, 1), (3, 0), (3, 2), (4, 1), (5, 3)] {
            let energy = -1.0 / (2.0 * (n * n) as f64);
            // Amplitude scale for this orbital.
            let amp = (0..4000)
                .map(|i| radial_unchecked(n, l, 0.01 * i as f64).abs())
                .fold(0.0_f64, f64::max);
            assert!(amp > 0.0);
            for &r in &[0.4_f64, 1.0, 2.0, 5.0, 12.0] {
                let rp = r + h;
                let rm = r - h;
                if rm <= 0.0 {
                    continue;
                }
                let f = |x: f64| radial_unchecked(n, l, x);
                let (vp, vm, v0) = (f(rp), f(rm), f(r));
                let dp = (vp - vm) / (2.0 * h);
                let d2p = (vp - 2.0 * v0 + vm) / (h * h);
                let lf = l as f64;
                let potential = lf * (lf + 1.0) / (2.0 * r * r) - 1.0 / r;
                let residual = (-0.5 * d2p - dp / r + potential * v0) - energy * v0;
                assert!(
                    residual.abs() < 1e-3 * amp,
                    "n={n} l={l} r={r} residual={residual} amp={amp}"
                );
            }
        }
    }

    #[test]
    fn normalization_factor_matches_closed_forms() {
        assert!(rel(normalization_factor(1, 0), 2.0) < 1e-14);
        assert!(rel(normalization_factor(2, 0), 1.0 / (2.0 * SQRT_2)) < 1e-14);
        assert!(rel(normalization_factor(2, 1), 1.0 / (2.0 * 6.0_f64.sqrt())) < 1e-14);
    }

    #[test]
    fn supported_range_constant_is_documented_and_enforced_by_test() {
        // The largest n in the validated range must still be finite, smooth and
        // normalisable.
        let n = MAX_VALIDATED_N;
        let l = n - 1;
        let v = radial_wavefunction(n, l, 5.0).unwrap();
        assert!(v.is_finite());
        let total = simpson(
            |r| r * r * radial_unchecked(n, l, r).powi(2),
            0.0,
            8.0 * (n * n) as f64,
            400_000,
        );
        assert!((total - 1.0).abs() < 1e-6, "n={n} l={l} total={total}");
    }

    /// Radial nodal structure: `R_{nl}` must have exactly `n - l - 1` interior
    /// sign changes.
    #[test]
    fn radial_node_count_is_n_minus_l_minus_one() {
        for n in 1..=6u32 {
            for l in 0..n {
                let nodes = n - l - 1;
                let r_max = 4.0 * (n * n) as f64 + 20.0;
                let steps = 400_000;
                let dr = r_max / steps as f64;
                let mut sign_changes = 0;
                let mut prev_sign = 0i32;
                for i in 1..steps {
                    let r = i as f64 * dr;
                    let v = radial_unchecked(n, l, r);
                    let s = if v > 0.0 {
                        1
                    } else if v < 0.0 {
                        -1
                    } else {
                        0
                    };
                    if s != 0 {
                        if prev_sign != 0 && s != prev_sign {
                            sign_changes += 1;
                        }
                        prev_sign = s;
                    }
                }
                assert_eq!(sign_changes, nodes, "n={n} l={l}");
            }
        }
    }
}
