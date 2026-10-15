//! Numerical methods and special functions.
//!
//! This crate is deliberately free of any physics-domain knowledge beyond the
//! mathematical special functions needed to evaluate hydrogenic wavefunctions.
//! It has no dependency on graphics, GPU, or Python bindings.

pub mod quadrature;
pub mod sampling;
pub mod special;

/// Common numerical tolerances used across validation tests.
///
/// The values are documented so that every tolerance in the test suite refers
/// to a single, explicit definition rather than a magic number.
pub mod tolerances {
    /// Relative tolerance for special-function evaluations compared against
    /// closed-form analytical expressions (double precision, well-conditioned).
    pub const SPECIAL_FUNCTION_REL: f64 = 1e-12;

    /// Relative tolerance for composite Simpson quadrature of smooth integrands
    /// with a sufficiently fine grid.
    pub const QUADRATURE_REL: f64 = 1e-10;

    /// Absolute tolerance for orthonormality residuals of spherical harmonics.
    pub const ORTHONORMALITY_ABS: f64 = 1e-8;

    /// Absolute tolerance for the radial normalisation integral
    /// `\int_0^inf r^2 R_{nl}(r)^2 dr = 1`.
    pub const NORMALIZATION_ABS: f64 = 1e-8;

    /// Relative tolerance for the numerical residual of the radial
    /// Schrodinger equation evaluated with a finite-difference Laplacian.
    pub const SCHRODINGER_RESIDUAL_REL: f64 = 1e-4;
}
