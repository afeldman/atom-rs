//! Unit system and physical constants.
//!
//! # Convention (normative)
//!
//! **All internal calculations are performed in Hartree atomic units.**
//! In this system:
//!
//! | quantity      | atomic unit | value in SI              |
//! |---------------|-------------|--------------------------|
//! | length        | bohr `a0`   | 5.29177210903e-11 m      |
//! | energy        | hartree `E_h` | 4.3597447222071e-18 J  |
//! | mass          | electron mass `m_e` | 9.1093837015e-31 kg |
//! | charge        | elementary charge `e` | 1.602176634e-19 C |
//! | action        | `hbar`      | 1.054571817e-34 J s      |
//! | time          | `hbar/E_h`  | 2.4188843265857e-17 s    |
//!
//! This means that with `a0 = 1`, `m_e = 1`, `e = 1`, `hbar = 1`, and
//! `1/(4 pi eps0) = 1`, the hydrogen Hamiltonian reads
//! `H = -1/2 laplacian - 1/r` and the ground-state energy is exactly `-0.5`.
//!
//! Every public function that consumes or returns a dimensioned quantity states
//! the unit in its documentation. Conversion helpers are provided in
//! [`conversions`]; they are the *only* place in the crate where non-atomic
//! units are used.
//!
//! Values are CODATA 2018.

use serde::{Deserialize, Serialize};

/// Physical constants in SI units (CODATA 2018).
pub mod constants {
    /// Bohr radius `a0` in metres.
    pub const BOHR_RADIUS_M: f64 = 5.291_772_109_03e-11;
    /// Hartree energy in joules.
    pub const HARTREE_J: f64 = 4.359_744_722_207_1e-18;
    /// Hartree energy in electronvolts.
    pub const HARTREE_EV: f64 = 27.211_386_245_988;
    /// Hartree energy in kJ/mol.
    pub const HARTREE_KJ_PER_MOL: f64 = 2625.4996394799;
    /// Electron mass in kilograms.
    pub const ELECTRON_MASS_KG: f64 = 9.109_383_701_5e-31;
    /// Proton mass in kilograms.
    pub const PROTON_MASS_KG: f64 = 1.672_621_923_69e-27;
    /// Unified atomic mass unit in kilograms.
    pub const AMU_KG: f64 = 1.660_539_066_60e-27;
    /// Unified atomic mass unit in electron masses.
    pub const AMU_IN_ELECTRON_MASSES: f64 = 1822.888486209;
    /// Atomic unit of time in seconds.
    pub const ATOMIC_TIME_S: f64 = 2.418_884_326_585_7e-17;
    /// Speed of light in atomic units (dimensionless, `c = 1/alpha`).
    pub const SPEED_OF_LIGHT_AU: f64 = 137.035_999_084;
}

/// Length unit for API input/output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LengthUnit {
    /// Bohr radius (`a0`). This is the internal unit.
    #[default]
    Bohr,
    /// Angstrom (`1e-10 m`).
    Angstrom,
    /// Nanometre (`1e-9 m`).
    Nanometer,
}

impl LengthUnit {
    /// Multiplicative factor converting a value *from* this unit *to* bohr.
    pub const fn to_bohr_factor(self) -> f64 {
        match self {
            LengthUnit::Bohr => 1.0,
            LengthUnit::Angstrom => 1.0 / (constants::BOHR_RADIUS_M * 1e10),
            LengthUnit::Nanometer => 1.0 / (constants::BOHR_RADIUS_M * 1e9),
        }
    }

    /// Convert a value expressed in this unit into bohr.
    pub fn to_bohr(self, value: f64) -> f64 {
        value * self.to_bohr_factor()
    }

    /// Convert a value expressed in bohr into this unit.
    pub fn from_bohr(self, bohr: f64) -> f64 {
        bohr / self.to_bohr_factor()
    }
}

/// Energy unit for API input/output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum EnergyUnit {
    /// Hartree (`E_h`). This is the internal unit.
    #[default]
    Hartree,
    /// Electronvolt.
    ElectronVolt,
    /// Kilojoule per mole.
    KilojoulePerMole,
    /// Joule.
    Joule,
}

impl EnergyUnit {
    /// Multiplicative factor converting a value *from* this unit *to* hartree.
    pub const fn to_hartree_factor(self) -> f64 {
        match self {
            EnergyUnit::Hartree => 1.0,
            EnergyUnit::ElectronVolt => 1.0 / constants::HARTREE_EV,
            EnergyUnit::KilojoulePerMole => 1.0 / constants::HARTREE_KJ_PER_MOL,
            EnergyUnit::Joule => 1.0 / constants::HARTREE_J,
        }
    }

    /// Convert a value expressed in this unit into hartree.
    pub fn to_hartree(self, value: f64) -> f64 {
        value * self.to_hartree_factor()
    }

    /// Convert a value expressed in hartree into this unit.
    pub fn from_hartree(self, hartree: f64) -> f64 {
        hartree / self.to_hartree_factor()
    }
}

/// Explicit conversion helpers between SI and atomic units.
pub mod conversions {
    use super::constants::*;

    /// Convert a length in metres to bohr.
    pub fn metre_to_bohr(m: f64) -> f64 {
        m / BOHR_RADIUS_M
    }

    /// Convert a length in bohr to metres.
    pub fn bohr_to_metre(b: f64) -> f64 {
        b * BOHR_RADIUS_M
    }

    /// Convert a length in angstrom to bohr.
    pub fn angstrom_to_bohr(a: f64) -> f64 {
        a * 1e-10 / BOHR_RADIUS_M
    }

    /// Convert a length in bohr to angstrom.
    pub fn bohr_to_angstrom(b: f64) -> f64 {
        b * BOHR_RADIUS_M / 1e-10
    }

    /// Convert an energy in hartree to electronvolts.
    pub fn hartree_to_ev(h: f64) -> f64 {
        h * HARTREE_EV
    }

    /// Convert an energy in electronvolts to hartree.
    pub fn ev_to_hartree(ev: f64) -> f64 {
        ev / HARTREE_EV
    }

    /// Convert an energy in hartree to joules.
    pub fn hartree_to_joule(h: f64) -> f64 {
        h * HARTREE_J
    }

    /// Convert an energy in joules to hartree.
    pub fn joule_to_hartree(j: f64) -> f64 {
        j / HARTREE_J
    }

    /// Convert an energy in hartree to kJ/mol.
    pub fn hartree_to_kj_per_mol(h: f64) -> f64 {
        h * HARTREE_KJ_PER_MOL
    }

    /// Convert a mass in atomic mass units to atomic units of mass (`m_e`).
    pub fn amu_to_electron_mass(u: f64) -> f64 {
        u * AMU_IN_ELECTRON_MASSES
    }

    /// Convert a mass in atomic units of mass (`m_e`) to atomic mass units.
    pub fn electron_mass_to_amu(m: f64) -> f64 {
        m / AMU_IN_ELECTRON_MASSES
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_length_units() {
        for &v in &[0.0, 0.5, 1.0, 3.75, 100.0] {
            let b = LengthUnit::Angstrom.to_bohr(v);
            let back = LengthUnit::Angstrom.from_bohr(b);
            assert!((back - v).abs() < 1e-12 * v.abs().max(1.0));
        }
    }

    #[test]
    fn one_bohr_is_expected_angstrom() {
        // 1 a0 = 0.529177210903 A
        let a = LengthUnit::Angstrom.from_bohr(1.0);
        assert!((a - 0.529_177_210_903).abs() < 1e-12, "a={a}");
        // and the inverse
        assert!((LengthUnit::Angstrom.to_bohr(a) - 1.0).abs() < 1e-14);
        // Bohr is the identity
        assert_eq!(LengthUnit::Bohr.from_bohr(1.0), 1.0);
        assert_eq!(LengthUnit::Bohr.to_bohr(1.0), 1.0);
    }

    #[test]
    fn hartree_to_ev_reference_value() {
        assert!((conversions::hartree_to_ev(1.0) - 27.211_386_245_988).abs() < 1e-12);
        // The classic textbook value
        assert!((conversions::hartree_to_ev(1.0) - 27.2114).abs() < 1e-3);
    }

    #[test]
    fn hydrogen_ground_state_energy_conversions() {
        // -0.5 Ha = -13.605693122994 eV
        let ha = -0.5_f64;
        assert!((conversions::hartree_to_ev(ha) + 13.605_693_122_994).abs() < 1e-11);
        // -0.5 Ha = -1312.7498197 kJ/mol
        assert!((conversions::hartree_to_kj_per_mol(ha) + 1312.74981973995).abs() < 1e-8);
    }

    #[test]
    fn round_trip_energy_units() {
        for &v in &[-0.5, -0.125, 1.0, 13.6] {
            let h = EnergyUnit::ElectronVolt.to_hartree(v);
            let back = EnergyUnit::ElectronVolt.from_hartree(h);
            assert!((back - v).abs() < 1e-12 * v.abs().max(1.0));
        }
    }

    #[test]
    fn amu_round_trip() {
        // proton mass ~ 1836.15267344 m_e
        let m = conversions::amu_to_electron_mass(1.007_276_466_88);
        assert!((m - 1836.15267344).abs() < 1e-4, "m={m}");
        let back = conversions::electron_mass_to_amu(m);
        assert!((back - 1.007_276_466_88).abs() < 1e-12);
    }

    #[test]
    fn speed_of_light_atomic_units_matches_inverse_fine_structure() {
        // c = 1/alpha with alpha = 7.2973525693e-3
        let c = 1.0 / 7.297_352_569_3e-3;
        assert!((c - constants::SPEED_OF_LIGHT_AU).abs() < 1e-2);
    }
}
