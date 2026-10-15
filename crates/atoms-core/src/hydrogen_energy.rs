//! Hydrogen Energy Eigenvalues
//! 
//! The energy eigenvalues for hydrogen are given by:
//! E_n = -0.5 / n² Hartree = -27.211396 eV / n²
//! 
//! where n = 1, 2, 3, ... (principal quantum number)

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EnergyLevel {
    pub n: u32,
    pub hartree: f64,
    pub eV: f64,
}

impl EnergyLevel {
    /// Calculate energy eigenvalue for given principal quantum number.
    /// 
    /// Formula: E_n = -0.5 / n² Hartree
    pub fn new(n: u32) -> Self {
        let hartree = -0.5 / (n as f64 * n as f64);
        // 1 Hartree = 27.211396 eV
        let eV = hartree * 27.211396;
        
        Self { n, hartree, eV }
    }

    /// Ground state energy (n=1)
    pub fn ground() -> Self {
        Self::new(1)
    }

    /// Verify against analytical solution E_n = -0.5/n² Ha
    pub fn verify_analytical(&self) -> bool {
        let expected = -0.5 / (self.n as f64 * self.n as f64);
        // Use relative tolerance for floating-point comparison
        const REL_TOL: f64 = 1e-12;
        (self.hartree - expected).abs() / expected.abs() < REL_TOL
    }

    pub fn to_joules(&self) -> f64 {
        // 1 Hartree = 4.359744722e-18 J
        self.hartree * 4.359744722e-18
    }

    pub fn to_kj_mol(&self) -> f64 {
        // Convert Hartree to kJ/mol
        // 1 Hartree = 2625.4996 kJ/mol (approximately)
        self.hartree * 2625.4996
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ground_state_energy() {
        let energy = EnergyLevel::ground();
        assert_eq!(energy.n, 1);
        assert!((energy.hartree - (-0.5)).abs() < 1e-14);
        assert!((energy.eV - (-27.211396)).abs() < 0.0001);
    }

    #[test]
    fn test_excited_states() {
        let energy_2 = EnergyLevel::new(2);
        assert!((energy_2.hartree - (-0.125)).abs() < 1e-14);
        
        let energy_3 = EnergyLevel::new(3);
        assert!((energy_3.hartree - (-0.055556)).abs() < 1e-14);
    }

    #[test]
    fn test_analytical_verification() {
        for n in 1..=10 {
            let energy = EnergyLevel::new(n);
            assert!(energy.verify_analytical(), "n={}", n);
        }
    }
}
