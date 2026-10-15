//! # atoms-orbitals
//!
//! Hydrogen-like (one-electron) wavefunctions in position space.
//!
//! This crate implements the **exact analytical solutions** of the
//! time-independent Schrodinger equation for a single electron in the Coulomb
//! potential of a nucleus of charge `Z`:
//!
//! ```text
//! psi_{nlm}(r, theta, phi) = R_{nl}(r) * Y_l^m(theta, phi)
//! ```
//!
//! with the normalised radial function
//!
//! ```text
//! R_{nl}(r) = sqrt( (2/n)^3 (n-l-1)! / (2n (n+l)!) )
//!             * exp(-rho/2) * rho^l * L_{n-l-1}^{2l+1}(rho),
//!   rho = 2 r / n
//! ```
//!
//! and the complex spherical harmonics `Y_l^m` defined in
//! [`spherical_harmonics`].
//!
//! ## Normative conventions
//!
//! * Atomic units (bohr, hartree). See [`atoms_core::units`].
//! * `psi` is normalised as `int |psi|^2 dV = 1` over `R^3`.
//! * Angles use the **physics** convention: `theta` is the polar angle from
//!   `+z` in `[0, pi]`, `phi` is the azimuth from `+x` in `[0, 2*pi)`.
//! * `Y_l^m` includes the Condon-Shortley phase `(-1)^m` for `m > 0` and obeys
//!   `Y_l^{-m} = (-1)^m conj(Y_l^m)`.
//! * Negative `m` is fully supported and is **not** the same as `|m|`: the two
//!   differ by a complex conjugate and, for odd `m`, by a sign.
//!
//! ## Probability density versus radial probability density
//!
//! These are different quantities and the crate keeps them apart explicitly:
//!
//! | quantity | symbol | meaning | unit |
//! |---|---|---|---|
//! | probability density | `|psi|^2` | probability per unit volume, integrated over `R^3` gives 1 | `1/bohr^3` |
//! | radial probability density | `P(r) = r^2 |R_{nl}(r)|^2` | probability per unit *radius*, integrated over `r >= 0` gives 1 | `1/bohr` |
//!
//! `r^2 |R|^2` is the density of the radial *measure*, obtained after
//! integrating the angular part out; it is **not** `|psi|^2` sampled on a
//! sphere, and `|psi|^2` at the origin is *not* zero for s orbitals while
//! `P(0)` is zero for every orbital.
//!
//! ## Scope and limitations
//!
//! * One electron, one nucleus, no screening, no electron-electron interaction,
//!   no spin, no relativistic corrections, no finite nuclear size.
//! * Infinite nuclear mass (see [`atoms_core::hydrogen_energy::reduced_mass_ratio`]
//!   for the correction factor, which is not applied automatically).
//! * Applying `Z > 1` gives the *unscreened* hydrogenic approximation, which is
//!   not a description of a multi-electron atom.

pub mod density;
pub mod expectations;
pub mod quantum_numbers;
pub mod radial;
pub mod sampling;
pub mod spherical_harmonics;
pub mod wavefunction;

pub use density::{radial_expectation_value, DensityGrid, DensityGridError};
pub use quantum_numbers::{QuantumNumberError, QuantumNumbers};
pub use radial::{
    radial_probability_density, radial_probability_density_z, radial_wavefunction,
    radial_wavefunction_z,
};
pub use sampling::{OrbitalSampler, SamplerError};
pub use spherical_harmonics::{
    spherical_harmonic, spherical_harmonic_abs_squared, SphericalHarmonic,
};
pub use wavefunction::HydrogenWavefunction;
