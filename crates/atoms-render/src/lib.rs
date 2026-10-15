//! # atoms-render
//!
//! The portable rendering foundation: it turns a scalar field on a regular grid
//! into an image, on the CPU always and on the GPU when a `wgpu` adapter is
//! available.
//!
//! ## Independence from the scientific core
//!
//! This crate **depends on no physics crate**, and with default features it has
//! **no dependencies at all**. The two sides meet at [`VolumeData`], a plain
//! description of a grid and its values; see that type for the axis-order and
//! unit contract.
//!
//! The direction of the coupling is fixed: the viewer and any bridge code depend
//! on both sides, but neither `atoms-core` nor `atoms-orbitals` may depend on
//! this crate. Rendering is replaceable; the physics is not.
//!
//! ## What is and is not accelerated
//!
//! * **Rendering can run on the GPU** through the `gpu` feature. It is *not*
//!   described as accelerated: measured against the CPU reference on the Apple M4
//!   Max, the GPU path is **slower** for the image sizes tested, because each
//!   offscreen render pays for a synchronised readback. See
//!   `docs/rendering.md` §12 for the numbers and the reason.
//! * **Scientific computation is not.** All physics is computed by the CPU
//!   crates, and the renderer consumes the result. There is no GPU compute kernel
//!   in this milestone, so no scientific speed-up is claimed anywhere.
//!
//! ## Layers
//!
//! | module | contents | needs a GPU |
//! |---|---|---|
//! | [`volume`] | [`VolumeData`], [`TransferFunction`], axis description | no |
//! | [`cpu`] | reference implementation of every render | no |
//! | [`image`] | [`RgbaImage`], the comparison report, a minimal PNG writer | no |
//! | [`error`] | [`RenderError`] and its classification | no |
//! | [`gpu`] | `wgpu` context, resources, slice/MIP pipelines, readback | yes |
//!
//! ## Resource ownership
//!
//! GPU objects follow one rule: **the context owns the device and queue, and
//! every resource owns what it creates**. [`gpu::GpuContext`] holds an `Arc` to
//! the device and queue; a pipeline holds the bind-group layout it was built
//! from; a render holds the buffers and the texture it allocates for that call
//! and drops them when it returns. Nothing borrows from a temporary, so no
//! resource can outlive its owner. Staging buffers are created per readback and
//! released immediately after the mapped data has been copied out.
//!
//! See `docs/rendering.md` for the full ownership table.
//!
//! ## Graceful degradation
//!
//! [`gpu::init_blocking`] and [`gpu::request_context`] return
//! [`RenderError::NoAdapter`] on a machine with no usable GPU, and
//! [`RenderError::UnsupportedCapability`] or [`RenderError::ResourceLimit`] when
//! the hardware cannot do what was asked. [`RenderError::is_unavailable`] lets a
//! caller fall back to [`cpu`] without treating "no GPU here" as a failure.
//!
//! ## Example (CPU only, no graphics stack needed)
//!
//! ```no_run
//! use atoms_render::{cpu, SliceAxis, TransferFunction, VolumeData};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let values = vec![1.0_f32; 4 * 4 * 4];
//! let volume = VolumeData::new([4, 4, 4], [-1.0; 3], [0.5, 0.5, 0.5], values, "bohr^-3")?;
//!
//! let tf = TransferFunction::log_scaled(volume.max());
//! let slice = cpu::render_slice(&volume, SliceAxis::Z, 2, &tf)?;
//! slice.write_png("/tmp/slice.png")?;
//! # Ok(())
//! # }
//! ```

pub mod cpu;
pub mod error;
pub mod image;
pub mod volume;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use cpu::{render_mip, render_mip_auto, render_slice, render_slice_auto};
pub use error::RenderError;
pub use image::{viridis, CompareReport, RgbaImage};
pub use volume::{SliceAxis, TransferFunction, VolumeData};

#[cfg(feature = "gpu")]
pub use gpu::{BackendInfo, ContextOptions, GpuContext};
