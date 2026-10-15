//! The full hydrogen-like wavefunction `psi_{nlm}(r, theta, phi)`.
//!
//! `psi = R_{nl}(r) * Y_l^m(theta, phi)` with `R` from [`crate::radial`] and
//! `Y` from [`crate::spherical_harmonics`]. The product is normalised so that
//!
//! ```text
//! int_{R^3} |psi|^2 dV = 1
//! ```
//!
//! Note that `psi` is in general **complex**: for `m != 0` the azimuthal factor
//! `e^{i m phi}` is complex. The imaginary part is a physical phase, not an
//! artefact; `|psi|^2` is real and non-negative for every `(n, l, m)`.
//!
//! # Angular nodal structure of complex orbitals
//!
//! Complex spherical harmonics are eigenfunctions of `L_z`, and their nodal
//! sets differ from the familiar "p_x, p_y, p_z" pictures, which are *real
//! linear combinations*:
//!
//! * `m = 0`: nodal surfaces are cones or planes perpendicular to `z`; for
//!   `l = 1` the node is the `z = 0` plane.
//! * `m = +-1`: the density vanishes on the `z` axis (a nodal *line* of
//!   azimuthal origin), not on a plane.
//! * `m = +-2`: likewise vanishes on the `z` axis, with `sin^2(theta)` weight.
//!
//! Real orbitals such as `p_x` are not implemented in M1 and are not silently
//! substituted.

use crate::quantum_numbers::{QuantumNumberError, QuantumNumbers};
use crate::radial::{radial_unchecked, radial_wavefunction_z};
use crate::spherical_harmonics::{evaluate as ylm, SphericalHarmonic};
use atoms_core::coordinates::cartesian_to_spherical;
use atoms_core::EnergyLevel;
use num_complex::Complex;

/// A validated hydrogen-like orbital.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HydrogenWavefunction {
    qn: QuantumNumbers,
    z: u32,
}

impl HydrogenWavefunction {
    /// Hydrogen (`Z = 1`) orbital with the given quantum numbers.
    ///
    /// # Errors
    /// Returns [`QuantumNumberError`] if `n == 0`, `l >= n`, or `|m| > l`.
    pub fn new(n: u32, l: u32, m: i32) -> Result<Self, QuantumNumberError> {
        Self::hydrogenic(n, l, m, 1)
    }

    /// One-electron ion of nuclear charge `z`.
    ///
    /// `z > 1` is the **unscreened** hydrogenic approximation and is not a
    /// description of a multi-electron atom. See the crate documentation.
    ///
    /// # Errors
    /// Returns [`QuantumNumberError`] for invalid quantum numbers, or if
    /// `z == 0`.
    pub fn hydrogenic(n: u32, l: u32, m: i32, z: u32) -> Result<Self, QuantumNumberError> {
        if z == 0 {
            return Err(QuantumNumberError::PrincipalMustBePositive);
        }
        Ok(Self {
            qn: QuantumNumbers::new(n, l, m)?,
            z,
        })
    }

    /// The validated quantum numbers.
    pub fn quantum_numbers(&self) -> QuantumNumbers {
        self.qn
    }

    /// Nuclear charge `Z`.
    pub fn z(&self) -> u32 {
        self.z
    }

    /// Analytic energy eigenvalue in hartree.
    pub fn energy(&self) -> EnergyLevel {
        EnergyLevel::new(self.qn.n(), self.z).expect("validated quantum numbers")
    }

    /// Designation such as `"2p"` or `"2p (m=-1)"`.
    pub fn designation(&self) -> String {
        self.qn.designation()
    }

    /// `psi(r, theta, phi)` in bohr^{-3/2}.
    ///
    /// `theta` is the polar angle from `+z`, `phi` the azimuth from `+x`.
    pub fn value_spherical(&self, r: f64, theta: f64, phi: f64) -> Complex<f64> {
        if r < 0.0 {
            return Complex::new(f64::NAN, f64::NAN);
        }
        let n = self.qn.n();
        let l = self.qn.l();
        let rad = if self.z == 1 {
            radial_unchecked(n, l, r)
        } else {
            // radial_wavefunction_z cannot fail here: quantum numbers validated.
            radial_wavefunction_z(n, l, self.z, r).expect("validated quantum numbers")
        };
        let ang = ylm(l, self.qn.m(), theta, phi);
        ang * rad
    }

    /// `psi(x, y, z)` in bohr^{-3/2}, with Cartesian input in bohr.
    pub fn value_cartesian(&self, x: f64, y: f64, z: f64) -> Complex<f64> {
        let s = cartesian_to_spherical(x, y, z);
        self.value_spherical(s.r, s.theta, s.phi)
    }

    /// `|psi(r, theta, phi)|^2` in bohr^{-3}.
    ///
    /// This is the probability *density*; it is dimensionless only after
    /// multiplication by a volume element. It is **not** the radial probability
    /// density `r^2 |R|^2`; see the crate documentation.
    pub fn probability_density_spherical(&self, r: f64, theta: f64, phi: f64) -> f64 {
        self.value_spherical(r, theta, phi).norm_sqr()
    }

    /// `|psi(x, y, z)|^2` in bohr^{-3}, Cartesian input in bohr.
    pub fn probability_density(&self, x: f64, y: f64, z: f64) -> f64 {
        self.value_cartesian(x, y, z).norm_sqr()
    }

    /// Radial probability density `P(r) = r^2 |R_{nl}(r)|^2` in bohr^{-1}.
    ///
    /// Satisfies `int_0^inf P(r) dr = 1` and is zero at `r = 0` for every
    /// orbital.
    pub fn radial_probability_density(&self, r: f64) -> f64 {
        let rad = self.radial_value(r);
        r * r * rad * rad
    }

    /// The radial wavefunction `R_{nl}(r)` alone, in bohr^{-3/2}.
    ///
    /// Provided so that callers can build radial integrands directly, for
    /// example `int_0^inf r^k R(r)^2 dr`, without having to divide by `r^2` and
    /// thereby lose the `r = 0` behaviour of `P(r)`.
    pub fn radial_value(&self, r: f64) -> f64 {
        if r < 0.0 {
            return f64::NAN;
        }
        if self.z == 1 {
            radial_unchecked(self.qn.n(), self.qn.l(), r)
        } else {
            radial_wavefunction_z(self.qn.n(), self.qn.l(), self.z, r)
                .expect("validated quantum numbers")
        }
    }

    /// Angular probability density `|Y_l^m(theta)|^2` in sr^{-1}, independent of
    /// `phi`.
    pub fn angular_probability_density(&self, theta: f64) -> f64 {
        SphericalHarmonic::new(self.qn.l(), self.qn.m())
            .expect("validated quantum numbers")
            .abs_squared(theta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    fn close(a: f64, b: f64, tol: f64, ctx: &str) {
        assert!((a - b).abs() < tol, "{ctx}: {a} vs {b}");
    }

    #[test]
    fn ground_state_matches_closed_form() {
        // psi_100 = (1/sqrt(pi)) e^{-r}
        let wf = HydrogenWavefunction::new(1, 0, 0).unwrap();
        for &r in &[0.0, 0.5, 1.0, 3.0] {
            for &theta in &[0.0, 1.0, PI] {
                let got = wf.value_spherical(r, theta, 0.0);
                assert!(got.im.abs() < 1e-15);
                close(
                    got.re,
                    (-r).exp() / PI.sqrt(),
                    1e-14,
                    &format!("r={r} theta={theta}"),
                );
            }
        }
    }

    #[test]
    fn two_s_matches_closed_form() {
        // psi_200 = (1/(4 sqrt(2 pi))) (2 - r) e^{-r/2}
        let wf = HydrogenWavefunction::new(2, 0, 0).unwrap();
        let expect = |r: f64| (2.0 - r) * (-r / 2.0).exp() / (4.0 * (2.0 * PI).sqrt());
        for &r in &[0.0_f64, 0.5, 2.0, 5.0, 9.0] {
            close(wf.value_spherical(r, 0.7, 0.2).re, expect(r), 1e-14, "2s");
        }
    }

    #[test]
    fn two_p_m_zero_matches_closed_form() {
        // psi_210 = (1/(4 sqrt(2 pi))) r e^{-r/2} cos(theta)
        let wf = HydrogenWavefunction::new(2, 1, 0).unwrap();
        for &r in &[0.5_f64, 2.0, 6.0] {
            for &theta in &[0.0, 1.1, PI] {
                let expect = r * (-r / 2.0).exp() * theta.cos() / (4.0 * (2.0 * PI).sqrt());
                close(wf.value_spherical(r, theta, 0.0).re, expect, 1e-14, "2p0");
            }
        }
    }

    #[test]
    fn complex_phase_appears_only_for_nonzero_m() {
        let wf0 = HydrogenWavefunction::new(3, 2, 0).unwrap();
        assert!(wf0.value_spherical(2.0, 1.0, 2.5).im.abs() < 1e-15);

        let wf1 = HydrogenWavefunction::new(3, 2, 1).unwrap();
        assert!(wf1.value_spherical(2.0, 1.0, 0.0).im.abs() < 1e-15);
        // at phi = pi/2 the phase is e^{i pi/2} = i, so the value is imaginary
        let v = wf1.value_spherical(2.0, 1.0, FRAC_PI_2);
        assert!(v.re.abs() < 1e-14, "re={}", v.re);
        assert!(v.im.abs() > 1e-6, "im={}", v.im);
    }

    #[test]
    fn negative_and_positive_m_have_equal_magnitude() {
        for (n, l) in [(2u32, 1u32), (3, 2), (4, 3)] {
            for m in 1..=l as i32 {
                let a = HydrogenWavefunction::new(n, l, m).unwrap();
                let b = HydrogenWavefunction::new(n, l, -m).unwrap();
                for &(r, t, p) in &[(1.0, 0.8, 0.3), (3.0, 2.1, 4.0)] {
                    close(
                        a.probability_density_spherical(r, t, p),
                        b.probability_density_spherical(r, t, p),
                        1e-14,
                        "|m| symmetry",
                    );
                }
            }
        }
    }

    #[test]
    fn cartesian_and_spherical_agree() {
        use atoms_core::coordinates::spherical_to_cartesian;
        let wf = HydrogenWavefunction::new(3, 2, -1).unwrap();
        for &(r, t, p) in &[(0.7, 0.4, 1.2), (2.5, 2.0, 5.5), (6.0, 1.0, 3.3)] {
            let cart = spherical_to_cartesian(r, t, p);
            let a = wf.value_spherical(r, t, p);
            let b = wf.value_cartesian(cart.x, cart.y, cart.z);
            assert!((a - b).norm() < 1e-13, "{a} vs {b}");
        }
    }

    #[test]
    fn probability_density_is_non_negative_and_factorises() {
        let wf = HydrogenWavefunction::new(4, 2, -2).unwrap();
        for &(r, t, p) in &[(0.5, 0.3, 0.0), (3.0, 1.4, 2.2), (12.0, 2.9, 5.0)] {
            let rho = wf.probability_density_spherical(r, t, p);
            assert!(rho >= 0.0, "rho={rho}");
            let factorised = {
                let rad = crate::radial::radial_unchecked(4, 2, r);
                rad * rad * wf.angular_probability_density(t)
            };
            close(rho, factorised, 1e-15, "factorisation");
        }
    }

    #[test]
    fn radial_probability_density_is_zero_at_origin_and_scales_correctly() {
        let wf = HydrogenWavefunction::new(1, 0, 0).unwrap();
        assert_eq!(wf.radial_probability_density(0.0), 0.0);
        // P(r) = r^2 * (2 e^{-r})^2 = 4 r^2 e^{-2r}; maximum at r = 1 (bohr)
        let at_one = wf.radial_probability_density(1.0);
        close(at_one, 4.0 * (-2.0f64).exp(), 1e-14, "P(1)");
        // stationary point at r = 1
        let h = 1e-6;
        let d = (wf.radial_probability_density(1.0 + h) - wf.radial_probability_density(1.0 - h))
            / (2.0 * h);
        assert!(d.abs() < 1e-6, "dP/dr = {d}");
    }

    /// `m = 0` p orbitals have the `z = 0` plane as a node; `|m| = 1` p orbitals
    /// vanish on the `z` axis instead.
    #[test]
    fn angular_nodes_distinguish_m_zero_from_m_one() {
        let wf0 = HydrogenWavefunction::new(2, 1, 0).unwrap();
        // z = 0 plane: theta = pi/2
        for &phi in &[0.0, 1.0, 3.0] {
            close(
                wf0.probability_density_spherical(2.0, FRAC_PI_2, phi),
                0.0,
                1e-28,
                "m=0 node",
            );
        }
        assert!(wf0.probability_density_spherical(2.0, 0.0, 0.0) > 0.0);

        let wf1 = HydrogenWavefunction::new(2, 1, 1).unwrap();
        // z axis: theta = 0 or pi
        close(
            wf1.probability_density_spherical(2.0, 0.0, 0.0),
            0.0,
            1e-28,
            "m=1 axis node",
        );
        close(
            wf1.probability_density_spherical(2.0, PI, 0.0),
            0.0,
            1e-28,
            "m=1 axis node",
        );
        assert!(wf1.probability_density_spherical(2.0, FRAC_PI_2, 0.0) > 0.0);
    }

    #[test]
    fn energy_matches_analytic_level() {
        for (n, l, m) in [(1u32, 0u32, 0i32), (2, 1, -1), (4, 3, 2)] {
            let wf = HydrogenWavefunction::new(n, l, m).unwrap();
            close(
                wf.energy().hartree,
                -1.0 / (2.0 * (n * n) as f64),
                1e-15,
                "energy",
            );
            assert_eq!(wf.energy().n, n);
        }
        let he_plus = HydrogenWavefunction::hydrogenic(1, 0, 0, 2).unwrap();
        close(he_plus.energy().hartree, -2.0, 1e-15, "He+ 1s");
    }

    #[test]
    fn hydrogenic_z_shrinks_the_orbital_by_one_over_z() {
        let h = HydrogenWavefunction::new(1, 0, 0).unwrap();
        let he = HydrogenWavefunction::hydrogenic(1, 0, 0, 2).unwrap();
        // rho_Z(r) = Z^3 rho_1(Z r)
        let r = 0.3;
        close(
            he.probability_density(r, 0.0, 0.0),
            8.0 * h.probability_density(2.0 * r, 0.0, 0.0),
            1e-15,
            "Z scaling",
        );
    }

    #[test]
    fn invalid_construction_is_rejected() {
        assert!(HydrogenWavefunction::new(0, 0, 0).is_err());
        assert!(HydrogenWavefunction::new(1, 1, 0).is_err());
        assert!(HydrogenWavefunction::new(2, 1, 2).is_err());
        assert!(HydrogenWavefunction::new(2, 1, -2).is_err());
        assert!(HydrogenWavefunction::hydrogenic(1, 0, 0, 0).is_err());
    }

    #[test]
    fn angular_probability_density_integrates_to_one_over_two_pi() {
        use atoms_numerics::quadrature::simpson;
        // int_0^pi |Y|^2 sin(theta) dtheta = 1 / (2 pi), because the full
        // normalisation int dphi int dtheta gives 1 and the phi integral is 2 pi.
        let expected = 1.0 / (2.0 * PI);
        for (l, m) in [(0u32, 0i32), (1, 0), (1, 1), (2, -2), (3, 3), (4, -1)] {
            let wf = HydrogenWavefunction::new(l + 1, l, m).unwrap();
            let total = simpson(
                |t| wf.angular_probability_density(t) * t.sin(),
                0.0,
                PI,
                200_000,
            );
            assert!(
                (total - expected).abs() < 1e-9,
                "l={l} m={m} total={total} expected={expected}"
            );
        }
    }

    /// The product of the two marginal densities must reconstruct the full
    /// normalisation: `int_0^inf P(r) dr * int_0^pi |Y|^2 sin(theta) dtheta
    /// * 2 pi = 1`.
    #[test]
    fn marginal_densities_reconstruct_unit_normalisation() {
        use atoms_numerics::quadrature::simpson;
        for (n, l, m) in [(1u32, 0u32, 0i32), (2, 1, -1), (3, 2, 2)] {
            let wf = HydrogenWavefunction::new(n, l, m).unwrap();
            let r_max = 12.0 * (n * n) as f64 + 30.0;
            let radial = simpson(|r| wf.radial_probability_density(r), 0.0, r_max, 200_000);
            let angular = simpson(
                |t| wf.angular_probability_density(t) * t.sin(),
                0.0,
                PI,
                200_000,
            );
            let total = radial * angular * 2.0 * PI;
            assert!(
                (total - 1.0).abs() < 1e-8,
                "n={n} l={l} m={m} total={total}"
            );
        }
    }
}
