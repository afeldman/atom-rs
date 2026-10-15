//! # atoms-gpu (not yet implemented)
//!
//! GPU compute kernels for scientific workloads.
//!
//! **Status: scheduled for M5.** No kernel exists yet. The GPU backend is
//! optional by design: `atoms-core`, `atoms-numerics`, and `atoms-orbitals`
//! build and run with no graphics hardware and no GPU toolchain.
//!
//! ## Architectural contract (fixed in M0, to be honoured in M5)
//!
//! * Rendering lives in `atoms-render`; compute lives here. They may share a
//!   device, but they are separate crates with separate interfaces and separate
//!   resource lifetimes.
//! * Every accelerated kernel must have a CPU reference implementation in the
//!   corresponding CPU crate, and a test asserting agreement within a
//!   documented tolerance and reporting the measured maximum error.
//! * Performance claims require reproducible measurements, including kernel
//!   time, transfer time, and peak memory — never a speedup number without the
//!   measurement that produced it.
//!
//! ## Backends
//!
//! * `wgpu` compute for portable acceleration (Vulkan/Metal/DX12).
//! * CUDA behind the `cuda` feature for NVIDIA-specific kernels, always with a
//!   CPU fallback so the project remains buildable without an NVIDIA toolchain.
//!
//! ## Explicit non-goal
//!
//! No assumption that every numerical method benefits from GPU acceleration. A
//! kernel is only added once profiling shows that it is a bottleneck.

/// The milestone in which the first kernel in this crate is scheduled.
pub const IMPLEMENTED_IN_MILESTONE: &str = "M5";
