//! Atoms Core Library
//! 
//! Provides fundamental quantum mechanical calculations for:
//! - Hydrogen atom energy eigenvalues
//! - Radial wavefunctions R_{n,l}(r)
//! - Spherical harmonics Y_l^m(θ,φ)
//! - Probability densities |ψ|²
//! 
//! All calculations are validated against analytical solutions.

pub mod units;
pub mod coordinates;
pub mod hydrogen_energy;
pub mod radial_wavefunction;
pub mod spherical_harmonics;
pub mod wavefunction;
