//! Unit system for quantum calculations
//! 
//! We use atomic units (Hartree) as the default internal representation,
//! with explicit conversion functions to SI units and other common units.

use serde::{Deserialize, Serialize};

/// Unit of length: options include bohr (a₀), angstrom, nm, pm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LengthUnit {
    Bohr,      // 1 a₀ = 5.29177210903e-11 m (atomic unit)
    Angstrom,  // 1 Å = 1e-10 m
    Nanometer, // 1 nm = 1e-9 m
    Picometer, // 1 pm = 1e-12 m
}

impl Default for LengthUnit {
    fn default() -> Self {
        Self::Bohr
    }
}

/// Unit of energy: options include Hartree (atomic), eV, Joule
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnergyUnit {
    Hartree,   // 1 Ha = 4.359744722e-18 J (atomic unit)
    ElectronVolt,
    Joule,
}

impl Default for EnergyUnit {
    fn default() -> Self {
        Self::Hartree
    }
}

/// Conversion factors to SI units (base reference is atomic units)
pub mod conversions {
    // Length: atomic units (bohr radius a₀) to SI
    pub const BOHR_TO_SI: f64 = 5.29177210903e-11;  // meters
    
    // Energy: Hartree to SI
    pub const HARTREE_TO_SI: f64 = 4.359744722e-18; // Joules
    
    // Energy: eV to atomic (Hartree)
    pub const EV_TO_HARTREE: f64 = 1.0 / 27.211396;
    
    // Time: Hartree atomic units to SI
    pub const HARTREE_TIME_TO_SI: f64 = 2.418884326585e-17; // seconds
    
    // Frequency: Hartree to Hz
    pub const HARTREE_TO_HZ: f64 = 4.134137333e15;

    pub fn bohr_to_angstrom(value: f64) -> f64 {
        value / BOHR_TO_SI * 1e-10
    }
    
    pub fn hartree_to_ev(value: f64) -> f64 {
        value * 27.211396
    }

    pub fn joule_to_hartree(value: f64) -> f64 {
        value / HARTREE_TO_SI
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypothesis_energy;

    #[test]
    fn test_conversion_consistency() {
        let energy = EnergyLevel::ground();
        let hartree = energy.hartree;
        
        // Hartree to eV conversion
        let ev = hartree / conversions::EV_TO_HARTREE;
        assert!((ev - (-27.211396)).abs() < 0.001);
    }
}
