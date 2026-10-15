//! Atomic species and atom instances.
//!
//! # Scope and honest limitations
//!
//! This module provides **descriptive** atomic data: symbol, nuclear charge,
//! nuclear mass, and the nuclear position. It exists so that molecules can be
//! represented and so that the one-electron (hydrogenic) model can be applied
//! to species with `Z > 1`.
//!
//! It does **not** implement multi-electron atomic structure. In particular:
//!
//! * `nominal_configuration` is human-readable reference metadata taken from
//!   standard tables. It is not used by any calculation in this crate, it is not
//!   derived from a self-consistent field, and it must not be presented as a
//!   computed electronic structure.
//! * Screening, electron-electron repulsion, exchange and correlation are
//!   absent. Applying the hydrogenic wavefunctions of `atoms-orbitals` to a
//!   species with `Z > 1` yields an *unscreened one-electron* approximation
//!   whose accuracy degrades rapidly with `Z`.
//!
//! Any function that uses the hydrogenic model is named accordingly, and no
//! function in this module claims to compute a total atomic energy.

use nalgebra::Point3;
use serde::{Deserialize, Serialize};

/// A neutral atomic species.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AtomicSpecies {
    /// IUPAC chemical symbol, e.g. `"H"`.
    pub symbol: String,
    /// Nuclear charge `Z` (atomic number).
    pub z: u32,
    /// Nuclear mass in electron masses (`m_e`).
    pub mass_electron_masses: f64,
    /// Nominal ground-state configuration, descriptive metadata only.
    ///
    /// **Not** used by any calculation. See the module documentation.
    pub nominal_configuration: String,
}

impl AtomicSpecies {
    /// The hydrogen atom, `Z = 1`, proton mass.
    pub fn hydrogen() -> Self {
        Self {
            symbol: "H".to_string(),
            z: 1,
            mass_electron_masses: crate::units::constants::PROTON_MASS_KG
                / crate::units::constants::ELECTRON_MASS_KG,
            nominal_configuration: "1s1".to_string(),
        }
    }

    /// Look up a species by atomic number for `1 <= Z <= 10`.
    ///
    /// Returns `None` outside that range. The table is intentionally small and
    /// explicit rather than generated, so that every entry is auditable.
    pub fn from_z(z: u32) -> Option<Self> {
        // (symbol, mass in u, nominal configuration)
        let (symbol, mass_u, config) = match z {
            1 => ("H", 1.007_825_031_9, "1s1"),
            2 => ("He", 4.002_602_2, "1s2"),
            3 => ("Li", 6.94, "[He] 2s1"),
            4 => ("Be", 9.012_183_1, "[He] 2s2"),
            5 => ("B", 10.81, "[He] 2s2 2p1"),
            6 => ("C", 12.011, "[He] 2s2 2p2"),
            7 => ("N", 14.007, "[He] 2s2 2p3"),
            8 => ("O", 15.999, "[He] 2s2 2p4"),
            9 => ("F", 18.998_403_163, "[He] 2s2 2p5"),
            10 => ("Ne", 20.1797, "[He] 2s2 2p6"),
            _ => return None,
        };
        Some(Self {
            symbol: symbol.to_string(),
            z,
            mass_electron_masses: crate::units::conversions::amu_to_electron_mass(mass_u),
            nominal_configuration: config.to_string(),
        })
    }

    /// Nuclear mass in atomic mass units.
    pub fn mass_amu(&self) -> f64 {
        crate::units::conversions::electron_mass_to_amu(self.mass_electron_masses)
    }

    /// Whether the unscreened one-electron (hydrogenic) model is exact for this
    /// species. True only for hydrogen.
    pub fn hydrogenic_model_is_exact(&self) -> bool {
        self.z == 1
    }
}

/// An atom placed in space.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Atom {
    /// The species.
    pub species: AtomicSpecies,
    /// Nuclear position in bohr.
    pub position: Point3<f64>,
}

impl Atom {
    /// A hydrogen atom at the origin.
    pub fn hydrogen() -> Self {
        Self {
            species: AtomicSpecies::hydrogen(),
            position: Point3::origin(),
        }
    }

    /// An atom of the given species at the given position (bohr).
    pub fn new(species: AtomicSpecies, position: Point3<f64>) -> Self {
        Self { species, position }
    }

    /// Nuclear charge `Z`.
    pub fn z(&self) -> u32 {
        self.species.z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hydrogen_mass_is_about_1836_electron_masses() {
        let h = AtomicSpecies::hydrogen();
        assert!((h.mass_electron_masses - 1836.15267344).abs() < 1e-4);
        assert!((h.mass_amu() - 1.007_276_466_88).abs() < 1e-9);
    }

    #[test]
    fn species_table_covers_z_one_to_ten() {
        for z in 1..=10u32 {
            let s = AtomicSpecies::from_z(z).expect("table entry");
            assert_eq!(s.z, z);
            assert!(!s.symbol.is_empty());
            assert!(s.mass_electron_masses > 0.0);
        }
        assert!(AtomicSpecies::from_z(0).is_none());
        assert!(AtomicSpecies::from_z(11).is_none());
    }

    #[test]
    fn species_table_symbols_and_z_are_consistent() {
        let expected = [
            (1, "H"),
            (2, "He"),
            (3, "Li"),
            (4, "Be"),
            (5, "B"),
            (6, "C"),
            (7, "N"),
            (8, "O"),
            (9, "F"),
            (10, "Ne"),
        ];
        for (z, sym) in expected {
            assert_eq!(AtomicSpecies::from_z(z).unwrap().symbol, sym);
        }
    }

    #[test]
    fn masses_increase_monotonically_with_z() {
        let mut prev = 0.0;
        for z in 1..=10u32 {
            let m = AtomicSpecies::from_z(z).unwrap().mass_electron_masses;
            assert!(m > prev, "z={z} m={m} prev={prev}");
            prev = m;
        }
    }

    #[test]
    fn only_hydrogen_is_exact_in_the_one_electron_model() {
        assert!(AtomicSpecies::hydrogen().hydrogenic_model_is_exact());
        assert!(!AtomicSpecies::from_z(2)
            .unwrap()
            .hydrogenic_model_is_exact());
    }

    #[test]
    fn atom_construction_keeps_position_and_charge() {
        let a = Atom::new(
            AtomicSpecies::from_z(6).unwrap(),
            Point3::new(1.0, 2.0, 3.0),
        );
        assert_eq!(a.z(), 6);
        assert_eq!(a.position, Point3::new(1.0, 2.0, 3.0));
        assert_eq!(Atom::hydrogen().z(), 1);
    }
}
