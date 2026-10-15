//! Atoms Core Library
//! 
//! Provides fundamental quantum mechanical calculations for:
//! - Hydrogen atom energy eigenvalues
//! - Radial wavefunctions R_{n,l}(r)
//! - Spherical harmonics Y_l^m(θ,φ)
//! - Probability densities |ψ|²
//! 
//! All calculations are validated against analytical solutions.

pub mod hydrogen_energy;
pub mod spatial_grid;
pub mod units;
