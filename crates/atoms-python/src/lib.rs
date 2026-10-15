//! # atoms-python (not yet implemented)
//!
//! PyO3 bindings exposing the Rust core as the Python package `atoms`.
//!
//! **Status: scheduled for M2.** The `python` feature is present but no binding
//! is registered yet. Building with `--features python` produces an extension
//! module with an empty module body.
//!
//! ## Contract (fixed in M0, to be honoured in M2)
//!
//! * The Rust crates must never depend on this crate. The dependency direction
//!   is `Python -> atoms-python -> atoms-core/atoms-orbitals`.
//! * Bulk data crosses the boundary as NumPy-compatible buffers without
//!   gratuitous copies.
//! * Invalid parameters raise idiomatic Python exceptions; they never panic and
//!   never abort the interpreter.
//! * Every result carries the unit it is expressed in. The Python API must not
//!   infer units from a parameter name.
//! * Random sampling is seeded explicitly and reproducibly.

/// The milestone in which the first binding is scheduled.
pub const IMPLEMENTED_IN_MILESTONE: &str = "M2";

#[cfg(feature = "python")]
mod bindings {
    use pyo3::prelude::*;

    /// Placeholder module. No functions are exported yet; see the crate docs.
    #[pymodule]
    fn atoms(_py: Python<'_>, _m: &PyModule) -> PyResult<()> {
        Ok(())
    }
}
