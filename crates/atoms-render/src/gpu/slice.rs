//! Rendering a volume as a slice or a maximum-intensity projection.
//!
//! # What is validated
//!
//! The scalar channel of the returned image is compared against
//! [`crate::cpu`], which is the reference implementation. The comparison is
//! meaningful because both sides:
//!
//! * read the same values in the same index order (public array order);
//! * use the same transfer-function formula;
//! * map image pixels to grid points with the same orientation rule.
//!
//! A test asserts that agreement for an asymmetric volume, which is the only
//! kind that can detect a transposition or an axis swap.
//!
//! # Resource ownership
//!
//! [`SliceRenderer`] owns its pipeline and bind-group layout. Each call allocates
//! the volume buffer, the uniform buffer, the render target and the readback
//! staging buffer, and drops them all on return. [`SliceRenderer::upload_volume`]
//! and [`SliceRenderer::render_with_buffer`] exist separately so that a caller
//! that renders many views of one volume can pay the upload once, and so that a
//! benchmark can measure upload and render cost independently.

use crate::error::RenderError;
use crate::image::{viridis, RgbaImage};
use crate::volume::{SliceAxis, TransferFunction, VolumeData};

use super::{align_row_bytes, GpuContext};

/// Render target format. `Rgba8Unorm` (not `-Srgb`) is used deliberately: no
/// automatic transfer-function encoding is applied, so the bytes in memory are
/// exactly the values the shader wrote and the scalar channel can be compared
/// against the CPU reference without an inverse transform.
pub const RENDER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// `wgpu` uniform buffer layout; must match `Uniforms` in `slice.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    shape: [u32; 4],
    fixed: [u32; 4],
    params: [f32; 4],
    out_size: [u32; 4],
}

const MODE_SLICE: u32 = 0;
const MODE_MIP: u32 = 1;

fn axis_code(axis: SliceAxis) -> u32 {
    match axis {
        SliceAxis::X => 0,
        SliceAxis::Y => 1,
        SliceAxis::Z => 2,
    }
}

/// Uploads a volume once and renders slices or projections of it.
pub struct SliceRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
}

impl std::fmt::Debug for SliceRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SliceRenderer")
            .field("format", &self.format)
            .finish_non_exhaustive()
    }
}

impl SliceRenderer {
    /// Build the pipelines for `ctx`.
    ///
    /// # Errors
    /// [`RenderError::ResourceLimit`] if the device cannot bind a storage buffer
    /// in a fragment shader.
    pub fn new(ctx: &GpuContext) -> Result<Self, RenderError> {
        Self::with_format(ctx, RENDER_FORMAT)
    }

    /// Build the pipelines for a specific render target format.
    ///
    /// # Errors
    /// As [`SliceRenderer::new`].
    pub fn with_format(ctx: &GpuContext, format: wgpu::TextureFormat) -> Result<Self, RenderError> {
        if ctx.limits().max_storage_buffers_per_shader_stage == 0 {
            return Err(RenderError::UnsupportedCapability {
                what: "a storage buffer bound in a fragment shader".to_string(),
                available: "max_storage_buffers_per_shader_stage = 0".to_string(),
            });
        }

        let device = ctx.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("atoms-render slice shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "slice.wgsl"
            ))),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atoms-render slice bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<Uniforms>() as u64
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("atoms-render slice pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("atoms-render slice pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            format,
        })
    }

    /// The render target format this renderer writes.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Upload a volume to a device buffer.
    ///
    /// The buffer is created with `COPY_SRC` as well as `STORAGE` so that a test
    /// can read the uploaded bytes back and confirm that the GPU's view of the
    /// data equals the CPU's.
    ///
    /// # Errors
    /// [`RenderError::ResourceLimit`] if the volume exceeds the device's
    /// storage-binding limit, or the underlying allocation fails.
    pub fn upload_volume(
        &self,
        ctx: &GpuContext,
        volume: &VolumeData,
    ) -> Result<wgpu::Buffer, RenderError> {
        let bytes = volume.byte_len() as u64;
        ctx.check_storage_binding_bytes(bytes, "volume")?;
        let buffer = ctx.create_buffer(
            "atoms-render volume",
            bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        )?;
        let data: &[u8] = bytemuck::cast_slice(volume.values());
        ctx.queue().write_buffer(&buffer, 0, data);
        Ok(buffer)
    }

    /// Render the slice of `volume` at `index` along `axis`.
    ///
    /// `vmax` in `tf` should normally be the volume maximum so that the image is
    /// comparable with the CPU reference, which uses the same transfer function.
    ///
    /// # Errors
    /// [`RenderError::OutOfRange`] for a bad index, or a resource error.
    pub fn render_slice(
        &self,
        ctx: &GpuContext,
        volume: &VolumeData,
        axis: SliceAxis,
        index: usize,
        tf: &TransferFunction,
    ) -> Result<RgbaImage, RenderError> {
        if index >= volume.axis_len(axis) {
            return Err(RenderError::OutOfRange {
                what: format!("slice index along {}", axis.name()),
                requested: index,
                limit: volume.axis_len(axis),
            });
        }
        let buffer = self.upload_volume(ctx, volume)?;
        self.render_with_buffer(ctx, volume, &buffer, axis, MODE_SLICE, index, tf)
    }

    /// Render a maximum-intensity projection along `axis`.
    ///
    /// # Errors
    /// A resource error, or [`RenderError::Internal`] if the volume is empty.
    pub fn render_mip(
        &self,
        ctx: &GpuContext,
        volume: &VolumeData,
        axis: SliceAxis,
        tf: &TransferFunction,
    ) -> Result<RgbaImage, RenderError> {
        let buffer = self.upload_volume(ctx, volume)?;
        self.render_with_buffer(ctx, volume, &buffer, axis, MODE_MIP, 0, tf)
    }

    /// Render using an already-uploaded volume buffer.
    ///
    /// `volume` must describe the same data that was uploaded to `buffer`; its
    /// shape and geometry are used for the pixel mapping and the image size.
    ///
    /// # Errors
    /// [`RenderError::ResourceLimit`] if the image exceeds the device limits, or
    /// [`RenderError::ReadbackFailed`] if the readback fails.
    #[allow(clippy::too_many_arguments)]
    pub fn render_with_buffer(
        &self,
        ctx: &GpuContext,
        volume: &VolumeData,
        buffer: &wgpu::Buffer,
        axis: SliceAxis,
        mode: u32,
        index: usize,
        tf: &TransferFunction,
    ) -> Result<RgbaImage, RenderError> {
        let (width, height) = volume.slice_dims(axis);
        let (width, height) = (width as u32, height as u32);
        ctx.check_image_dims(width, height)?;

        let device = ctx.device();
        let queue = ctx.queue();

        let [nx, ny, nz] = volume.shape();
        let uniforms = Uniforms {
            shape: [nx as u32, ny as u32, nz as u32, 0],
            fixed: [
                mode,
                axis_code(axis),
                index as u32,
                volume.axis_len(axis) as u32,
            ],
            params: [tf.vmax, tf.compression, 0.0, 0.0],
            out_size: [width, height, 0, 0],
        };
        let uniform_buffer = ctx.create_buffer(
            "atoms-render uniforms",
            std::mem::size_of::<Uniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        )?;
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("atoms-render target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atoms-render slice bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("atoms-render encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("atoms-render slice pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        let row_bytes = width * 4;
        let padded_row_bytes = align_row_bytes(row_bytes);
        let staging_size = (padded_row_bytes as u64) * (height as u64);
        let staging = ctx.create_buffer(
            "atoms-render staging",
            staging_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        )?;

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(Some(encoder.finish()));

        let pixels = read_mapped_rows(ctx, &staging, height, row_bytes, padded_row_bytes)?;
        RgbaImage::from_pixels(width as usize, height as usize, pixels)
    }
}

/// Map a staging buffer, copy the unpadded rows out of it, and unmap it.
///
/// The copy out of the mapped range is unavoidable: `unmap` invalidates the
/// view, so the rows cannot be borrowed past it. The bytes are copied once,
/// straight into the buffer the caller keeps.
fn read_mapped_rows(
    ctx: &GpuContext,
    staging: &wgpu::Buffer,
    rows: u32,
    row_bytes: u32,
    padded_row_bytes: u32,
) -> Result<Vec<u8>, RenderError> {
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    ctx.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| RenderError::ReadbackFailed {
            message: format!("device poll failed: {e:?}"),
        })?;

    match rx.try_recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            return Err(RenderError::ReadbackFailed {
                message: format!("map_async failed: {e:?}"),
            })
        }
        Err(e) => {
            return Err(RenderError::ReadbackFailed {
                message: format!("map_async callback never fired: {e:?}"),
            })
        }
    }

    let mut out = Vec::with_capacity(rows as usize * row_bytes as usize);
    {
        let view = slice
            .get_mapped_range()
            .map_err(|e| RenderError::ReadbackFailed {
                message: format!("get_mapped_range failed: {e:?}"),
            })?;
        for row in 0..rows as usize {
            let start = row * padded_row_bytes as usize;
            let end = start + row_bytes as usize;
            if end > view.len() {
                return Err(RenderError::Internal {
                    message: format!(
                        "readback row {row} would read {end} bytes from a {}-byte mapping",
                        view.len()
                    ),
                });
            }
            out.extend_from_slice(&view[start..end]);
        }
    }
    staging.unmap();
    Ok(out)
}

/// Read the whole contents of a device buffer back to the host.
///
/// Used to prove that what the GPU reads is what the CPU uploaded.
///
/// # Errors
/// [`RenderError::ResourceLimit`] for an oversized request, or a readback error.
pub fn read_buffer(
    ctx: &GpuContext,
    buffer: &wgpu::Buffer,
    len: u64,
) -> Result<Vec<u8>, RenderError> {
    let staging = ctx.create_buffer(
        "atoms-render buffer readback",
        len,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    )?;
    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("atoms-render buffer readback encoder"),
        });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, len);
    ctx.queue().submit(Some(encoder.finish()));

    read_mapped_rows(ctx, &staging, 1, len as u32, len as u32)
}

/// Decode a little-endian `f32` readback.
///
/// # Errors
/// [`RenderError::VolumeLength`] if the buffer length is not a multiple of four.
pub fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>, RenderError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(RenderError::VolumeLength {
            expected: bytes.len() / 4 * 4,
            got: bytes.len(),
        });
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

/// Compare the GPU's copy of a volume against the CPU-side data.
///
/// Returns the largest absolute difference. A correct upload gives exactly zero:
/// the bytes are copied verbatim and no arithmetic is involved.
///
/// # Errors
/// [`RenderError::VolumeLength`] if the lengths differ.
pub fn measure_upload_error(gpu: &[f32], cpu: &[f32]) -> Result<u32, RenderError> {
    if gpu.len() != cpu.len() {
        return Err(RenderError::VolumeLength {
            expected: cpu.len(),
            got: gpu.len(),
        });
    }
    let mut worst = 0u32;
    for (a, b) in gpu.iter().zip(cpu.iter()) {
        worst = worst.max(a.to_bits().abs_diff(b.to_bits()));
    }
    Ok(worst)
}

/// Render an image on the CPU that matches what the GPU shader produces.
///
/// Provided so that a caller can verify the GPU path against the reference
/// without reimplementing the convention; it delegates to [`crate::cpu`].
pub fn cpu_reference_slice(
    volume: &VolumeData,
    axis: SliceAxis,
    index: usize,
    tf: &TransferFunction,
) -> Result<RgbaImage, RenderError> {
    crate::cpu::render_slice(volume, axis, index, tf)
}

/// CPU reference for [`SliceRenderer::render_mip`].
pub fn cpu_reference_mip(
    volume: &VolumeData,
    axis: SliceAxis,
    tf: &TransferFunction,
) -> Result<RgbaImage, RenderError> {
    crate::cpu::render_mip(volume, axis, tf)
}

/// Compose a display pixel the way the shader does.
///
/// Kept here so that the colour-map convention has one definition on each side.
pub fn gpu_convention_pixel(scalar: u8) -> [u8; 4] {
    let rgb = viridis(scalar as f32 / 255.0);
    [rgb[0], rgb[1], rgb[2], scalar]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::ContextOptions;

    /// Get a context, or skip the test cleanly when the machine has no GPU.
    fn context() -> Option<GpuContext> {
        match super::super::init_blocking(ContextOptions::new().with_label("atoms-render test")) {
            Ok(ctx) => Some(ctx),
            Err(e) if e.is_unavailable() => {
                eprintln!("skipping GPU test: {e}");
                None
            }
            Err(e) => panic!("GPU initialisation failed unexpectedly: {e}"),
        }
    }

    /// Asymmetric in every axis, with values that encode their own coordinates,
    /// so a transposition or axis swap cannot go unnoticed.
    fn volume() -> VolumeData {
        let [nx, ny, nz] = [11usize, 7, 5];
        let mut values = Vec::new();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    values.push((i as f64 + 0.5 * j as f64 + 0.25 * k as f64) / (nx as f64));
                }
            }
        }
        VolumeData::from_f64(
            [nx, ny, nz],
            [-1.0, -0.5, 0.0],
            [0.1, 0.2, 0.3],
            &values,
            "bohr^-3",
            true,
        )
        .unwrap()
    }

    #[test]
    fn uniform_layout_matches_the_shader() {
        // vec4<u32>, vec4<u32>, vec4<f32>, vec4<u32> => 64 bytes, 16-byte aligned
        assert_eq!(std::mem::size_of::<Uniforms>(), 64);
        let u = Uniforms {
            shape: [0; 4],
            fixed: [0; 4],
            params: [0.0; 4],
            out_size: [0; 4],
        };
        assert_eq!(bytemuck::bytes_of(&u).len(), 64);
        // the fields must sit at the offsets the WGSL uniform layout expects
        let base = &u as *const Uniforms as usize;
        assert_eq!((&u.shape as *const _ as usize) - base, 0);
        assert_eq!((&u.fixed as *const _ as usize) - base, 16);
        assert_eq!((&u.params as *const _ as usize) - base, 32);
        assert_eq!((&u.out_size as *const _ as usize) - base, 48);
    }

    #[test]
    fn axis_codes_match_the_shader() {
        assert_eq!(axis_code(SliceAxis::X), 0);
        assert_eq!(axis_code(SliceAxis::Y), 1);
        assert_eq!(axis_code(SliceAxis::Z), 2);
    }

    #[test]
    fn decode_f32_rejects_misaligned_input() {
        assert_eq!(decode_f32(&[0, 0, 0, 0]).unwrap(), vec![0.0]);
        assert!(decode_f32(&[0, 0, 0]).is_err());
        let decoded = decode_f32(&1.5f32.to_le_bytes()).unwrap();
        assert_eq!(decoded, vec![1.5]);
    }

    #[test]
    fn upload_error_is_zero_for_identical_data() {
        assert_eq!(measure_upload_error(&[1.0, 2.0], &[1.0, 2.0]).unwrap(), 0);
        // one ULP difference in the last bit is detected
        let a = [1.0f32];
        let b = [f32::from_bits(1.0f32.to_bits() + 1)];
        assert_eq!(measure_upload_error(&a, &b).unwrap(), 1);
        assert!(measure_upload_error(&[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn cpu_reference_matches_the_cpu_module() {
        let v = volume();
        let tf = TransferFunction::log_scaled(v.max());
        let a = cpu_reference_slice(&v, SliceAxis::Z, 2, &tf).unwrap();
        let b = crate::cpu::render_slice(&v, SliceAxis::Z, 2, &tf).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn cpu_reference_agrees_across_all_axes_for_every_index() {
        // The exhaustive CPU-only check: the renderer's own conventions must be
        // self-consistent before any GPU is involved.
        let v = volume();
        let tf = TransferFunction::log_scaled(v.max());
        for axis in [SliceAxis::X, SliceAxis::Y, SliceAxis::Z] {
            for index in 0..v.axis_len(axis) {
                let img = cpu_reference_slice(&v, axis, index, &tf).unwrap();
                let (w, h) = v.slice_dims(axis);
                assert_eq!((img.width(), img.height()), (w, h));
            }
            let mip = cpu_reference_mip(&v, axis, &tf).unwrap();
            let (w, h) = v.slice_dims(axis);
            assert_eq!((mip.width(), mip.height()), (w, h));
        }
    }

    #[test]
    fn context_rejects_an_impossible_requirement_without_panicking() {
        let limits = wgpu::Limits {
            max_buffer_size: u64::MAX / 2,
            ..Default::default()
        };
        let options = ContextOptions::new()
            .with_required_limits(limits)
            .with_label("atoms-render impossible-limits test");
        match super::super::init_blocking(options) {
            Err(e) => assert!(
                e.is_capability_mismatch() || e.is_unavailable(),
                "unexpected error: {e}"
            ),
            Ok(_) => panic!("a device with u64::MAX/2 max_buffer_size cannot exist"),
        }
    }

    #[test]
    fn gpu_render_matches_cpu_reference_for_every_axis_and_index() {
        let Some(ctx) = context() else { return };
        let renderer = SliceRenderer::new(&ctx).unwrap();
        let v = volume();
        let tf = TransferFunction::log_scaled(v.max());

        for axis in [SliceAxis::X, SliceAxis::Y, SliceAxis::Z] {
            for index in 0..v.axis_len(axis) {
                let gpu = renderer.render_slice(&ctx, &v, axis, index, &tf).unwrap();
                let cpu = crate::cpu::render_slice(&v, axis, index, &tf).unwrap();
                assert_eq!(
                    (gpu.width(), gpu.height()),
                    (cpu.width(), cpu.height()),
                    "size mismatch for {axis:?} at {index}"
                );
                let report = gpu.compare_scalar(&cpu, 1).unwrap();
                assert!(
                    report.within_tolerance(),
                    "GPU/CPU mismatch for {axis:?} at index {index}: {report}"
                );
                let rgb = compare_rgb(&gpu, &cpu);
                assert!(
                    rgb <= 3,
                    "colour mismatch for {axis:?} at index {index}: max {rgb}"
                );
            }
        }
    }

    #[test]
    fn gpu_mip_matches_cpu_reference_for_every_axis() {
        let Some(ctx) = context() else { return };
        let renderer = SliceRenderer::new(&ctx).unwrap();
        let v = volume();
        let tf = TransferFunction::log_scaled(v.max());

        for axis in [SliceAxis::X, SliceAxis::Y, SliceAxis::Z] {
            let gpu = renderer.render_mip(&ctx, &v, axis, &tf).unwrap();
            let cpu = crate::cpu::render_mip(&v, axis, &tf).unwrap();
            let report = gpu.compare_scalar(&cpu, 1).unwrap();
            assert!(
                report.within_tolerance(),
                "GPU/CPU MIP mismatch for {axis:?}: {report}"
            );
        }
    }

    #[test]
    fn uploaded_volume_arrives_unchanged() {
        let Some(ctx) = context() else { return };
        let renderer = SliceRenderer::new(&ctx).unwrap();
        let v = volume();
        let buffer = renderer.upload_volume(&ctx, &v).unwrap();
        let bytes = read_buffer(&ctx, &buffer, v.byte_len() as u64).unwrap();
        let gpu = decode_f32(&bytes).unwrap();
        assert_eq!(gpu.len(), v.values().len());
        assert_eq!(
            measure_upload_error(&gpu, v.values()).unwrap(),
            0,
            "the GPU must receive exactly the bytes the CPU uploaded"
        );
    }

    #[test]
    fn rendering_is_deterministic_and_repeatable() {
        let Some(ctx) = context() else { return };
        let renderer = SliceRenderer::new(&ctx).unwrap();
        let v = volume();
        let tf = TransferFunction::log_scaled(v.max());
        let a = renderer
            .render_slice(&ctx, &v, SliceAxis::Z, 1, &tf)
            .unwrap();
        let b = renderer
            .render_slice(&ctx, &v, SliceAxis::Z, 1, &tf)
            .unwrap();
        assert_eq!(a.pixels(), b.pixels(), "two identical renders must agree");
    }

    #[test]
    fn out_of_range_slice_is_rejected_before_touching_the_gpu() {
        let Some(ctx) = context() else { return };
        let renderer = SliceRenderer::new(&ctx).unwrap();
        let v = volume();
        let tf = TransferFunction::log_scaled(v.max());
        match renderer.render_slice(&ctx, &v, SliceAxis::Z, 99, &tf) {
            Err(RenderError::OutOfRange {
                requested, limit, ..
            }) => {
                assert_eq!((requested, limit), (99, 5))
            }
            other => panic!("expected OutOfRange, got {other:?}"),
        }
    }

    #[test]
    fn an_oversized_volume_is_rejected_with_a_limit_error() {
        let Some(ctx) = context() else { return };
        let renderer = SliceRenderer::new(&ctx).unwrap();
        // Require a huge contiguous volume so the storage-binding check fires
        // before any allocation is attempted.
        let n = 4096usize;
        let shape = [n, n, n];
        let bytes = (n * n * n * 4) as u64;
        assert!(
            bytes > ctx.limits().max_storage_buffer_binding_size,
            "this test assumes a 4096^3 f32 volume exceeds the binding limit"
        );
        let v = VolumeData::new(shape, [0.0; 3], [1.0; 3], Vec::new(), "u");
        assert!(matches!(v, Err(RenderError::VolumeLength { .. })));
        match ctx.check_storage_binding_bytes(bytes, "volume") {
            Err(RenderError::ResourceLimit {
                limit,
                requested,
                available,
            }) => {
                assert!(limit.contains("storage"));
                assert_eq!(requested, bytes);
                assert!(available < bytes);
            }
            other => panic!("expected ResourceLimit, got {other:?}"),
        }
        let _ = renderer;
    }

    /// Largest per-channel RGB difference between two images.
    fn compare_rgb(a: &RgbaImage, b: &RgbaImage) -> u8 {
        let mut worst = 0u8;
        let pa_all = a.pixels().as_chunks::<4>().0;
        let pb_all = b.pixels().as_chunks::<4>().0;
        for (pa, pb) in pa_all.iter().zip(pb_all.iter()) {
            for c in 0..3 {
                worst = worst.max(pa[c].abs_diff(pb[c]));
            }
        }
        worst
    }
}
