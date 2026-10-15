//! Hydrogen-like energy eigenvalues.
//!
//! # Physics
//!
//! For a one-electron atom with nuclear charge `Z` (an unscreened
//! hydrogen-like ion), the bound-state energies of the time-independent
//! Schrodinger equation
//!
//! ```text
//! [-1/2 laplacian - Z/r] psi = E psi
//! ```
//!
//! in Hartree atomic units depend only on the principal quantum number `n`:
//!
//! ```text
//! E_n = -Z^2 / (2 n^2)   [hartree]
//! ```
//!
//! For `Z = 1` this gives `E_1 = -0.5` hartree `= -13.605693122994` eV and
//! `E_2 = -0.125` hartree, i.e. the Lyman-alpha transition energy
//! `E_2 - E_1 = 10.2042698422455` eV.
//!
//! The `1/n^2` degeneracy in `l` is exact only for the pure Coulomb potential
//! with no screening and no reduced-mass correction; see the limitations note
//! below.
//!
//! # Limitations
//!
//! * The finite nuclear mass is not accounted for here. A physically complete
//!   treatment replaces the electron mass by the reduced mass
//!   `mu = m_e M / (m_e + M)`, shifting `E_n` by a factor `mu/m_e`
//!   (~0.05 % for hydrogen). This module reports the infinite-nuclear-mass
//!   eigenvalues, which is the standard reference convention.
//! * Radiative corrections (Lamb shift, fine structure) are absent.

use crate::units::constants::HARTREE_EV;
use serde::{Deserialize, Serialize};

/// The analytically exact energy eigenvalues of a hydrogen-like ion.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EnergyLevel {
    /// Principal quantum number `n >= 1`.
    pub n: u32,
    /// Nuclear charge `Z >= 1`.
    pub z: u32,
    /// Energy in hartree.
    pub hartree: f64,
}

impl EnergyLevel {
    /// Energy of level `n` for nuclear charge `z`.
    ///
    /// Returns `None` for `n == 0` or `z == 0`, for which the formula is
    /// undefined (`n = 0` is not a valid principal quantum number).
    pub fn new(n: u32, z: u32) -> Option<Self> {
        if n == 0 || z == 0 {
            return None;
        }
        let nf = n as f64;
        let zf = z as f64;
        Some(Self {
            n,
            z,
            hartree: -(zf * zf) / (2.0 * nf * nf),
        })
    }

    /// Hydrogen ground state, `n = 1`, `Z = 1`: exactly `-0.5` hartree.
    pub fn hydrogen_ground() -> Self {
        Self {
            n: 1,
            z: 1,
            hartree: -0.5,
        }
    }

    /// Energy in electronvolts.
    pub fn ev(&self) -> f64 {
        self.hartree * HARTREE_EV
    }

    /// Energy in joules.
    pub fn joule(&self) -> f64 {
        crate::units::conversions::hartree_to_joule(self.hartree)
    }

    /// Energy in kJ/mol.
    pub fn kj_per_mol(&self) -> f64 {
        crate::units::conversions::hartree_to_kj_per_mol(self.hartree)
    }

    /// Energy expressed in the requested unit.
    pub fn in_unit(&self, unit: crate::units::EnergyUnit) -> f64 {
        unit.from_hartree(self.hartree)
    }

    /// Ionisation energy from this level to the continuum, in hartree.
    ///
    /// By definition `E_infinity = 0` in this convention, so this is
    /// `-E_n = Z^2 / (2 n^2)` and is always positive.
    pub fn ionisation_energy_hartree(&self) -> f64 {
        -self.hartree
    }

    /// Transition energy to level `n_final`, in hartree (negative for emission).
    pub fn transition_hartree(&self, n_final: u32) -> Option<f64> {
        EnergyLevel::new(n_final, self.z).map(|e| e.hartree - self.hartree)
    }

    /// Wavelength of the transition to `n_final` in nanometres, or `None` for a
    /// zero-energy (degenerate) transition.
    ///
    /// Uses `lambda = h c / |dE|` with `hc = 1239.841984` eV nm.
    ///
    /// **The returned wavelength uses the infinite-nuclear-mass energies.** The
    /// observed wavelengths of hydrogen lines are about 0.05 % longer, because
    /// the electron orbits the centre of mass with the reduced mass
    /// `mu = m_e M/(m_e + M)`. Use
    /// [`EnergyLevel::transition_wavelength_nm_with_reduced_mass`] to compare
    /// against measured spectra.
    pub fn transition_wavelength_nm(&self, n_final: u32) -> Option<f64> {
        let de_ev = self.transition_hartree(n_final)? * HARTREE_EV;
        if de_ev == 0.0 {
            return None;
        }
        const HC_EV_NM: f64 = 1239.841984332;
        Some(HC_EV_NM / de_ev.abs())
    }

    /// Wavelength in nanometres including the finite-nuclear-mass correction.
    ///
    /// `nuclear_mass_in_electron_masses` is the nuclear mass in units of the
    /// electron mass. For a proton this is `1836.15267344`, giving the reduced
    /// mass factor `mu/m_e = M/(M + 1) = 0.999455679...`.
    pub fn transition_wavelength_nm_with_reduced_mass(
        &self,
        n_final: u32,
        nuclear_mass_in_electron_masses: f64,
    ) -> Option<f64> {
        let lambda = self.transition_wavelength_nm(n_final)?;
        let r = reduced_mass_ratio(nuclear_mass_in_electron_masses);
        if r <= 0.0 {
            return None;
        }
        Some(lambda / r)
    }
}

/// Reduced-mass ratio `mu/m_e = M/(M + m_e)` for a nucleus of mass `M` given in
/// electron masses.
///
/// Scales every hydrogenic energy, and therefore the Rydberg constant, by this
/// factor. For hydrogen (`M = 1836.15267344 m_e`) the ratio is
/// `0.999455679424...`.
pub fn reduced_mass_ratio(nuclear_mass_in_electron_masses: f64) -> f64 {
    if nuclear_mass_in_electron_masses.is_nan() || nuclear_mass_in_electron_masses <= 0.0 {
        return f64::NAN;
    }
    nuclear_mass_in_electron_masses / (nuclear_mass_in_electron_masses + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::EnergyUnit;

    #[test]
    fn hydrogen_ground_state_is_exactly_minus_half() {
        let e = EnergyLevel::hydrogen_ground();
        assert_eq!(e.hartree, -0.5);
        // Analytic value: -0.5 * 27.211386245988 = -13.605693122994 eV
        assert!((e.ev() - (-13.605_693_122_994)).abs() < 1e-11);
    }

    /// Reference eigenvalues from the closed form. These are *derived*, so the
    /// test guards the implementation against transcription errors; the
    /// independent check of the same formula lives in the differential-equation
    /// and variational tests in `atoms-orbitals` and `tests/`.
    #[test]
    fn energies_follow_inverse_square_law() {
        let expected: [(u32, f64); 8] = [
            (1, -0.5),
            (2, -0.125),
            (3, -1.0 / 18.0),
            (4, -1.0 / 32.0),
            (5, -1.0 / 50.0),
            (6, -1.0 / 72.0),
            (7, -1.0 / 98.0),
            (8, -1.0 / 128.0),
        ];
        for (n, want) in expected {
            let e = EnergyLevel::new(n, 1).unwrap();
            assert!(
                (e.hartree - want).abs() < 1e-15,
                "n={n}: {} vs {want}",
                e.hartree
            );
        }
    }

    #[test]
    fn hydrogenic_scaling_is_z_squared() {
        for z in 1..=10u32 {
            let e = EnergyLevel::new(1, z).unwrap();
            let want = -0.5 * (z as f64).powi(2);
            assert!((e.hartree - want).abs() < 1e-13, "z={z}");
        }
    }

    #[test]
    fn invalid_quantum_numbers_are_rejected() {
        assert!(EnergyLevel::new(0, 1).is_none());
        assert!(EnergyLevel::new(1, 0).is_none());
        assert!(EnergyLevel::new(0, 0).is_none());
    }

    #[test]
    fn lyman_alpha_wavelength_infinite_nuclear_mass() {
        // Infinite-nuclear-mass convention, the one implemented by the closed
        // form E_n = -1/(2n^2): 1239.841984332 eV nm / 10.2042698422455 eV
        let e2 = EnergyLevel::new(2, 1).unwrap();
        let lambda = e2.transition_wavelength_nm(1).unwrap();
        assert!((lambda - 121.502_27).abs() < 0.001, "lambda = {lambda} nm");
    }

    #[test]
    fn lyman_alpha_wavelength_with_reduced_mass_matches_observation() {
        // Observed H Lyman-alpha (vacuum) = 121.567 nm. The infinite-mass value
        // is 0.05 % shorter; the reduced-mass correction closes the gap.
        let e2 = EnergyLevel::new(2, 1).unwrap();
        let lambda = e2
            .transition_wavelength_nm_with_reduced_mass(1, 1836.15267344)
            .unwrap();
        assert!(
            (lambda - 121.567).abs() < 0.005,
            "lambda = {lambda} nm (observed 121.567)"
        );
    }

    #[test]
    fn balmer_alpha_wavelength_matches_reference() {
        // Infinite-nuclear-mass: 656.1123 nm. Observed H-alpha (vacuum) is
        // 656.46 nm, i.e. the reduced-mass corrected value below.
        let e3 = EnergyLevel::new(3, 1).unwrap();
        let lambda = e3.transition_wavelength_nm(2).unwrap();
        assert!((lambda - 656.112_28).abs() < 0.002, "lambda = {lambda} nm");

        let corrected = e3
            .transition_wavelength_nm_with_reduced_mass(2, 1836.15267344)
            .unwrap();
        assert!(
            (corrected - 656.47).abs() < 0.02,
            "corrected = {corrected} nm (observed 656.46)"
        );
    }

    #[test]
    fn reduced_mass_ratio_is_below_one_and_increases_with_mass() {
        let r_h = reduced_mass_ratio(1836.15267344);
        assert!(r_h < 1.0 && r_h > 0.999);
        assert!((r_h - 0.999_455_679_425).abs() < 1e-11);
        // heavier nucleus -> ratio closer to 1
        assert!(reduced_mass_ratio(10_000.0) > r_h);
        assert!(reduced_mass_ratio(f64::INFINITY) == 1.0 || reduced_mass_ratio(1e18) > 0.999_999);
        assert!(reduced_mass_ratio(0.0).is_nan());
        assert!(reduced_mass_ratio(-5.0).is_nan());
    }

    #[test]
    fn ionisation_energy_is_positive_and_exceeds_transition() {
        let e1 = EnergyLevel::hydrogen_ground();
        assert!((e1.ionisation_energy_hartree() - 0.5).abs() < 1e-15);
        let e2 = EnergyLevel::new(2, 1).unwrap();
        assert!(e2.ionisation_energy_hartree() < e1.ionisation_energy_hartree());
    }

    #[test]
    fn transition_to_self_is_zero_and_has_no_wavelength() {
        let e2 = EnergyLevel::new(2, 1).unwrap();
        assert!((e2.transition_hartree(2).unwrap()).abs() < 1e-15);
        assert!(e2.transition_wavelength_nm(2).is_none());
    }

    #[test]
    fn unit_conversions_agree_with_shared_helpers() {
        let e = EnergyLevel::hydrogen_ground();
        assert!((e.in_unit(EnergyUnit::ElectronVolt) - e.ev()).abs() < 1e-12);
        assert!((e.joule() - crate::units::conversions::hartree_to_joule(-0.5)).abs() < 1e-30);
        // -0.5 Ha = -1312.74981974 kJ/mol
        assert!((e.kj_per_mol() + 1312.74981973995).abs() < 1e-8);
    }

    #[test]
    fn excited_states_are_less_bound_than_ground_state() {
        for n in 2..=10u32 {
            let e = EnergyLevel::new(n, 1).unwrap();
            assert!(e.hartree > -0.5, "n={n}");
            assert!(e.hartree < 0.0, "n={n}");
        }
    }
}
