//! # atoms-python
//!
//! PyO3 bindings exposing the Rust core as the Python extension module
//! `atoms._atoms`.
//!
//! The dependency direction is strictly `Python -> atoms-python -> atoms-core /
//! atoms-orbitals`. No Rust crate depends on this one, so the library remains
//! usable without a Python toolchain: the `python` feature is off by default and
//! `cargo test --workspace` never links against libpython.
//!
//! Layout:
//!
//! * [`ordering`] — pure Rust array-layout helpers, compiled and tested with
//!   the `python` feature **disabled**, so the NumPy axis-order contract is
//!   verified without needing Python.
//! * the extension module — argument validation, unit conversion, array
//!   marshalling, and exception mapping. It contains no physics.
//!
//! # Building
//!
//! ```text
//! cd crates/atoms-python
//! maturin develop --release        # into the active virtual environment
//! ```

pub mod ordering;

#[cfg(feature = "python")]
mod bindings;
