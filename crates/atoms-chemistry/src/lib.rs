//! # atoms-chemistry (not yet implemented)
//!
//! Electronic-structure solvers.
//!
//! **Status: scheduled for M4.** No solver exists yet. This crate must not be
//! used, and no function here may be described as producing a quantum-chemistry
//! result.
//!
//! ## Planned progression
//!
//! Each stage must be validated before the next begins:
//!
//! 1. Analytically tractable one-electron reference systems (already covered by
//!    `atoms-orbitals`).
//! 2. A minimal, fully documented H2 model with a stated two-centre basis and
//!    analytic integrals, validated against the documented literature value for
//!    the equilibrium bond length and binding energy.
//! 3. Restricted Hartree-Fock with a declared basis-set strategy (e.g. STO-3G),
//!    validated against published benchmark energies.
//! 4. SCF convergence diagnostics: iteration count, energy change, density
//!    change, and explicit failure reporting on non-convergence.
//! 5. Molecular orbitals and electron densities exported for visualisation.
//! 6. Density functional theory, only after RHF is validated.
//!
//! ## Required documentation per solver
//!
//! Hamiltonian, basis set, approximations, convergence criteria, supported
//! observables, and the reference values used for validation.

/// The milestone in which the first solver in this crate is scheduled.
pub const IMPLEMENTED_IN_MILESTONE: &str = "M4";
