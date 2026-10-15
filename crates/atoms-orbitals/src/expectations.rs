//! Analytic closed forms for hydrogenic expectation values.
//!
//! These are the **independent reference values** against which the numerical
//! machinery in [`crate::density`] is validated. They are exact results of the
//! Coulomb problem in atomic units with `Z = 1` and infinite nuclear mass:
//!
//! | quantity | closed form | unit |
//! |---|---|---|
//! | `<r>`    | `(3n^2 - l(l+1)) / 2` | bohr |
//! | `<r^2>`  | `n^2 (5n^2 + 1 - 3l(l+1)) / 2` | bohr^2 |
//! | `<1/r>`  | `1 / n^2` | bohr^-1 |
//! | `<1/r^2>`| `1 / (n^3 (l + 1/2))` | bohr^-2 |
//!
//! For a one-electron ion of charge `Z` the radial moments scale as
//! `<r^k>_Z = Z^{-k} <r^k>_{Z=1}`.
//!
//! These formulas are standard results of the Kramers-Pasternack relations.
//! They are reproduced here as literal expressions so that a test failure
//! points at the numerical implementation rather than at a shared abstraction.

/// Analytic `<r>` in bohr for hydrogen (`Z = 1`).
pub fn expectation_r(n: u32, l: u32) -> f64 {
    let nf = n as f64;
    let lf = l as f64;
    (3.0 * nf * nf - lf * (lf + 1.0)) / 2.0
}

/// Analytic `<r^2>` in bohr^2 for hydrogen (`Z = 1`).
pub fn expectation_r_squared(n: u32, l: u32) -> f64 {
    let nf = n as f64;
    let lf = l as f64;
    nf * nf * (5.0 * nf * nf + 1.0 - 3.0 * lf * (lf + 1.0)) / 2.0
}

/// Analytic `<1/r>` in bohr^-1 for hydrogen (`Z = 1`).
///
/// This equals `-2 E_n` by the virial theorem.
pub fn expectation_inverse_r(n: u32) -> f64 {
    1.0 / (n as f64 * n as f64)
}

/// Analytic `<1/r^2>` in bohr^-2 for hydrogen (`Z = 1`).
pub fn expectation_inverse_r_squared(n: u32, l: u32) -> f64 {
    1.0 / ((n as f64).powi(3) * (l as f64 + 0.5))
}

/// Analytic radial variance `<(r - <r>)^2>` in bohr^2.
pub fn radial_variance(n: u32, l: u32) -> f64 {
    let m1 = expectation_r(n, l);
    expectation_r_squared(n, l) - m1 * m1
}

/// Scale a `Z = 1` radial moment `<r^k>` to nuclear charge `Z`.
pub fn scale_to_z(value_at_z1: f64, k: i32, z: u32) -> f64 {
    value_at_z1 / (z as f64).powi(k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ground_state_values_match_hand_calculations() {
        // 1s: <r> = 3/2 a0, <r^2> = 3 a0^2, <1/r> = 1 a0^-1
        assert!((expectation_r(1, 0) - 1.5).abs() < 1e-15);
        assert!((expectation_r_squared(1, 0) - 3.0).abs() < 1e-15);
        assert!((expectation_inverse_r(1) - 1.0).abs() < 1e-15);
        assert!((radial_variance(1, 0) - 0.75).abs() < 1e-15);
    }

    #[test]
    fn two_p_values_match_hand_calculations() {
        // 2p: <r> = (3*4 - 2)/2 = 5 a0, <r^2> = 4(20+1-6)/2 = 30 a0^2
        assert!((expectation_r(2, 1) - 5.0).abs() < 1e-15);
        assert!((expectation_r_squared(2, 1) - 30.0).abs() < 1e-15);
        assert!((expectation_inverse_r(2) - 0.25).abs() < 1e-15);
        // <1/r^2> = 1/(8 * 1.5) = 1/12
        assert!((expectation_inverse_r_squared(2, 1) - 1.0 / 12.0).abs() < 1e-15);
    }

    #[test]
    fn two_s_across_a_shell_has_larger_radius_than_two_p() {
        // For fixed n, <r> decreases with l
        for n in 2..=6u32 {
            let mut prev = f64::INFINITY;
            for l in 0..n {
                let r = expectation_r(n, l);
                assert!(r < prev, "n={n} l={l}");
                prev = r;
            }
        }
    }

    /// Virial theorem for the Coulomb problem: `2<T> = -<V>` and
    /// `E = -<1/r>/... ` specifically `E_n = -<1/r>/2` in atomic units.
    #[test]
    fn virial_theorem_links_inverse_r_to_the_energy() {
        for n in 1..=8u32 {
            let e = -1.0 / (2.0 * (n * n) as f64);
            let virial = -expectation_inverse_r(n) / 2.0;
            assert!((e - virial).abs() < 1e-15, "n={n}");
        }
    }

    #[test]
    fn variance_is_positive_and_grows_with_n() {
        let mut prev = 0.0;
        for n in 1..=6u32 {
            let v = radial_variance(n, 0);
            assert!(v > 0.0, "n={n}");
            assert!(v > prev, "n={n}");
            prev = v;
        }
    }

    #[test]
    fn z_scaling_follows_inverse_powers() {
        // <1/r> scales as Z, <r> as 1/Z
        assert!((scale_to_z(1.0, -1, 2) - 2.0).abs() < 1e-15);
        assert!((scale_to_z(1.5, 1, 2) - 0.75).abs() < 1e-15);
        assert!((scale_to_z(3.0, 2, 2) - 0.75).abs() < 1e-15);
    }
}
