//! # atoms-molecule (not yet implemented)
//!
//! Molecular geometry representation and molecular properties.
//!
//! **Status: scheduled for M3.** Nothing in this crate computes a molecular
//! property yet. In particular, no total energy, binding energy, dipole moment,
//! vibrational frequency, electron density, or molecular orbital is available,
//! and none may be claimed.
//!
//! ## Planned scope (M3)
//!
//! * [`Molecule`]: a list of [`atoms_core::Atom`] plus charge and spin
//!   multiplicity, with geometric helpers (bond lengths, angles, centre of
//!   mass, moment-of-inertia tensor).
//! * Properties computed **only** when the selected model actually supports
//!   them; every property returns the model that produced it so that the
//!   provenance cannot be lost.
//! * Energy curves along a bond coordinate using the classical pair potentials
//!   from `atoms-interactions`, clearly labelled as approximate.
//!
//! ## Explicit non-goals until M4
//!
//! * No Hartree-Fock, no DFT, no basis sets, no self-consistent field.
//! * No claim that a classical potential reproduces a chemical bond.

/// The milestone in which this crate's functionality is scheduled.
pub const IMPLEMENTED_IN_MILESTONE: &str = "M3";
