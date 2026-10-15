//! # atoms-interactions (not yet implemented)
//!
//! Interatomic interaction potentials and their gradients.
//!
//! **Status: scheduled for M3.** No potential is implemented yet. This crate
//! exists so that the workspace layout and the dependency direction are fixed
//! before implementation:
//!
//! * `atoms-interactions` depends only on `atoms-core` and linear algebra.
//! * It must never depend on rendering, GPU, or Python bindings.
//!
//! ## Planned scope (M3)
//!
//! * An [`Interaction`] trait exposing `energy`, `force` (analytic gradient),
//!   and `energy_curve` for a pair of species at a given separation.
//! * Coulomb interaction (exact for point charges).
//! * Lennard-Jones 12-6 pair potential, explicitly labelled as a **classical,
//!   empirical** model with no claim to reproduce a chemical bond.
//! * A validation rule enforced by test: every analytic force must agree with a
//!   central finite-difference derivative of the corresponding energy to a
//!   documented tolerance.
//!
//! ## Explicit non-goals
//!
//! A classical pair potential is not a quantum-mechanical bond model. This
//! crate will never describe a covalent bond, and nothing here may be reported
//! as a quantum-chemistry result.

/// Marker for the crate's implementation status, so that callers cannot
/// accidentally assume the module is populated.
pub const IMPLEMENTED_IN_MILESTONE: &str = "M3";
