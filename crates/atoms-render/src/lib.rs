//! # atoms-render (not yet implemented)
//!
//! Portable GPU rendering built on `wgpu`.
//!
//! **Status: scheduled for M2.** No shader, pipeline, or buffer management
//! exists yet.
//!
//! ## Architectural contract (fixed in M0, to be honoured in M2)
//!
//! * `wgpu` is the default graphics abstraction, giving Vulkan on Linux and
//!   Windows and **Metal** on macOS and Apple Silicon. A native `ash` Vulkan
//!   backend is **not** introduced until profiling justifies it.
//! * Shaders are written in WGSL so that a single shader source serves every
//!   backend.
//! * This crate owns rendering resources only. All scientific computation comes
//!   from `atoms-core`/`atoms-orbitals`/`atoms-gpu` and is handed over as plain
//!   data, so the physics core never links against a graphics API.
//! * Backend selection and the resolved adapter must be reported at runtime, so
//!   that a rendered image can be attributed to a concrete GPU backend.

/// The milestone in which the first renderer is scheduled.
pub const IMPLEMENTED_IN_MILESTONE: &str = "M2";
