//! The portable GPU path (`wgpu`), compiled only with the `gpu` feature.
//!
//! # Scope
//!
//! This module accelerates **rendering only**. No physics is computed here: by
//! the time data reaches this module it is a [`crate::VolumeData`] produced by
//! the CPU crates. There is no compute kernel in this milestone, and therefore no
//! claim of GPU-accelerated scientific computation anywhere in this crate.
//!
//! # Threading
//!
//! [`request_context`] is asynchronous. [`init_blocking`] exists for callers that
//! cannot await; it blocks, so it is documented for use on a worker thread, not
//! on a UI thread. [`GpuContext`] is `Send + Sync` and can be moved to and shared
//! with worker threads.
//!
//! # Failure policy
//!
//! Nothing here panics because a capability is missing. Every rejection is a
//! typed [`RenderError`]:
//!
//! | situation | error |
//! |---|---|
//! | no adapter at all | [`RenderError::NoAdapter`] |
//! | only a software adapter, and it was disallowed | [`RenderError::NoAdapter`] |
//! | a required feature is missing | [`RenderError::UnsupportedCapability`] |
//! | a required limit exceeds the adapter's | [`RenderError::ResourceLimit`] |
//! | device creation failed | [`RenderError::DeviceCreationFailed`] |
//! | a buffer or image is too large for the device | [`RenderError::ResourceLimit`] |

pub mod slice;

pub use slice::SliceRenderer;

use crate::error::RenderError;
use std::sync::Arc;

/// How to obtain a GPU context.
#[derive(Debug, Clone)]
pub struct ContextOptions {
    /// Debug label for the instance and device.
    pub label: String,
    /// Backends to consider.
    pub backends: wgpu::Backends,
    /// Adapter power preference.
    pub power_preference: wgpu::PowerPreference,
    /// Whether a CPU/software adapter (e.g. `lavapipe`, `SwiftShader`) may be
    /// used. When `false` a software adapter is rejected as if none existed.
    pub allow_software_fallback: bool,
    /// Ask the backend for a fallback adapter explicitly.
    pub force_fallback_adapter: bool,
    /// Features the device must provide.
    pub required_features: wgpu::Features,
    /// Limits the device must provide. `None` means [`wgpu::Limits::default`],
    /// the conservative downlevel set that every conformant adapter supports.
    pub required_limits: Option<wgpu::Limits>,
}

impl Default for ContextOptions {
    fn default() -> Self {
        Self {
            label: "atoms-render".to_string(),
            backends: wgpu::Backends::all(),
            power_preference: wgpu::PowerPreference::HighPerformance,
            allow_software_fallback: true,
            force_fallback_adapter: false,
            required_features: wgpu::Features::empty(),
            required_limits: None,
        }
    }
}

impl ContextOptions {
    /// Default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Require a set of features.
    pub fn with_required_features(mut self, features: wgpu::Features) -> Self {
        self.required_features = features;
        self
    }

    /// Require a set of limits.
    pub fn with_required_limits(mut self, limits: wgpu::Limits) -> Self {
        self.required_limits = Some(limits);
        self
    }

    /// Reject software adapters.
    pub fn without_software_fallback(mut self) -> Self {
        self.allow_software_fallback = false;
        self
    }

    /// Restrict to a set of backends.
    pub fn with_backends(mut self, backends: wgpu::Backends) -> Self {
        self.backends = backends;
        self
    }

    /// Set the debug label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    fn effective_limits(&self) -> wgpu::Limits {
        self.required_limits.clone().unwrap_or_default()
    }
}

/// What the renderer actually got, for reporting and for provenance records.
///
/// This is a plain description with no `wgpu` types, so it can be logged,
/// serialised, or written into a calculation's metadata block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendInfo {
    /// Backend name: `"metal"`, `"vulkan"`, `"dx12"`, `"gl"`, or `"noop"`.
    pub backend: String,
    /// Adapter name as reported by the driver.
    pub adapter_name: String,
    /// `"discrete-gpu"`, `"integrated-gpu"`, `"virtual-gpu"`, `"cpu"`, or `"other"`.
    pub device_type: String,
    /// Driver name.
    pub driver: String,
    /// Driver version string.
    pub driver_info: String,
    /// PCI vendor id.
    pub vendor_id: u32,
    /// PCI device id.
    pub device_id: u32,
    /// Largest 2-D texture dimension supported.
    pub max_texture_dimension_2d: u32,
    /// Largest 3-D texture dimension supported.
    pub max_texture_dimension_3d: u32,
    /// Largest buffer supported.
    pub max_buffer_size: u64,
    /// Largest buffer that may be bound as a shader storage buffer.
    pub max_storage_buffer_binding_size: u64,
}

impl BackendInfo {
    /// Whether this is a software rasteriser rather than real hardware.
    pub fn is_software(&self) -> bool {
        self.device_type == "cpu"
    }

    /// One-line summary, e.g. `metal / Apple M4 Max (integrated-gpu)`.
    pub fn summary(&self) -> String {
        format!(
            "{} / {} ({})",
            self.backend, self.adapter_name, self.device_type
        )
    }

    /// Multi-line report for logs and validation documents.
    pub fn report(&self) -> String {
        format!(
            "backend:            {}\n\
             adapter:            {}\n\
             device type:        {}\n\
             driver:             {} ({})\n\
             vendor/device id:   {:#x}/{:#x}\n\
             max texture 2d/3d:  {}/{}\n\
             max buffer:         {} bytes\n\
             max storage binding:{} bytes",
            self.backend,
            self.adapter_name,
            self.device_type,
            self.driver,
            self.driver_info,
            self.vendor_id,
            self.device_id,
            self.max_texture_dimension_2d,
            self.max_texture_dimension_3d,
            self.max_buffer_size,
            self.max_storage_buffer_binding_size,
        )
    }
}

fn device_type_name(t: wgpu::DeviceType) -> &'static str {
    match t {
        wgpu::DeviceType::DiscreteGpu => "discrete-gpu",
        wgpu::DeviceType::IntegratedGpu => "integrated-gpu",
        wgpu::DeviceType::VirtualGpu => "virtual-gpu",
        wgpu::DeviceType::Cpu => "cpu",
        wgpu::DeviceType::Other => "other",
    }
}

fn backend_name(b: wgpu::Backend) -> String {
    format!("{b:?}").to_ascii_lowercase()
}

fn info_from(adapter: &wgpu::Adapter) -> BackendInfo {
    let a = adapter.get_info();
    let l = adapter.limits();
    BackendInfo {
        backend: backend_name(a.backend),
        adapter_name: a.name,
        device_type: device_type_name(a.device_type).to_string(),
        driver: a.driver,
        driver_info: a.driver_info,
        vendor_id: a.vendor,
        device_id: a.device,
        max_texture_dimension_2d: l.max_texture_dimension_2d,
        max_texture_dimension_3d: l.max_texture_dimension_3d,
        max_buffer_size: l.max_buffer_size,
        max_storage_buffer_binding_size: l.max_storage_buffer_binding_size,
    }
}

/// Reject a software adapter when one was not allowed.
///
/// Split out as a pure function over the adapter's description so the policy can
/// be tested without a GPU.
fn adapter_is_acceptable(
    device_type: wgpu::DeviceType,
    options: &ContextOptions,
) -> Result<(), RenderError> {
    if device_type == wgpu::DeviceType::Cpu && !options.allow_software_fallback {
        return Err(RenderError::NoAdapter {
            required_fallback: false,
        });
    }
    Ok(())
}

/// Check that every required feature is present.
///
/// Pure function: testable without a GPU.
pub fn check_required_features(
    required: wgpu::Features,
    available: wgpu::Features,
) -> Result<(), RenderError> {
    let missing = required - available;
    if !missing.is_empty() {
        return Err(RenderError::UnsupportedCapability {
            what: format!("required feature(s) {missing:?}"),
            available: format!("{available:?}"),
        });
    }
    Ok(())
}

/// Check that every required limit is at least the adapter's.
///
/// Only the limits this crate can actually consume are compared. Pure function:
/// testable without a GPU.
pub fn check_required_limits(
    required: &wgpu::Limits,
    available: &wgpu::Limits,
) -> Result<(), RenderError> {
    macro_rules! at_least {
        ($field:ident) => {
            if (required.$field as u64) > (available.$field as u64) {
                return Err(RenderError::ResourceLimit {
                    limit: stringify!($field).to_string(),
                    requested: required.$field as u64,
                    available: available.$field as u64,
                });
            }
        };
    }
    at_least!(max_buffer_size);
    at_least!(max_storage_buffer_binding_size);
    at_least!(max_storage_buffers_per_shader_stage);
    at_least!(max_texture_dimension_1d);
    at_least!(max_texture_dimension_2d);
    at_least!(max_texture_dimension_3d);
    at_least!(max_bind_groups);
    Ok(())
}

/// A device and queue plus a description of what they are.
///
/// Holds `Arc`s so that resources created from it can own what they need without
/// borrowing from the context.
#[derive(Clone)]
pub struct GpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    info: BackendInfo,
    limits: wgpu::Limits,
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

impl GpuContext {
    /// The device. Cloning the `Arc` keeps the device alive.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// The device limits in force.
    pub fn limits(&self) -> &wgpu::Limits {
        &self.limits
    }

    /// What this context is running on.
    pub fn info(&self) -> &BackendInfo {
        &self.info
    }

    /// Whether this context is a software rasteriser.
    pub fn is_software(&self) -> bool {
        self.info.is_software()
    }

    /// Verify a buffer size against the device limits before allocating.
    ///
    /// # Errors
    /// [`RenderError::ResourceLimit`] if the size or the binding is too large.
    pub fn check_buffer_bytes(&self, bytes: u64, what: &str) -> Result<(), RenderError> {
        if bytes > self.limits.max_buffer_size {
            return Err(RenderError::ResourceLimit {
                limit: format!("max_buffer_size ({what})"),
                requested: bytes,
                available: self.limits.max_buffer_size,
            });
        }
        Ok(())
    }

    /// Verify a buffer size against the *storage binding* limit.
    ///
    /// A buffer can be allocated while still being too large to bind as a
    /// storage buffer, which is a different limit and a different error.
    ///
    /// # Errors
    /// [`RenderError::ResourceLimit`] if the binding would be too large.
    pub fn check_storage_binding_bytes(&self, bytes: u64, what: &str) -> Result<(), RenderError> {
        if bytes > self.limits.max_storage_buffer_binding_size {
            return Err(RenderError::ResourceLimit {
                limit: format!("max_storage_buffer_binding_size ({what})"),
                requested: bytes,
                available: self.limits.max_storage_buffer_binding_size,
            });
        }
        Ok(())
    }

    /// Verify image dimensions against the device limits.
    ///
    /// # Errors
    /// [`RenderError::ResourceLimit`] if either dimension is too large.
    pub fn check_image_dims(&self, width: u32, height: u32) -> Result<(), RenderError> {
        let max = self.limits.max_texture_dimension_2d;
        if width > max || height > max {
            return Err(RenderError::ResourceLimit {
                limit: "max_texture_dimension_2d".to_string(),
                requested: width.max(height) as u64,
                available: max as u64,
            });
        }
        Ok(())
    }

    /// Create a buffer, validating its size first.
    ///
    /// # Errors
    /// [`RenderError::ResourceLimit`] if it would exceed a device limit.
    pub fn create_buffer(
        &self,
        label: &str,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> Result<wgpu::Buffer, RenderError> {
        self.check_buffer_bytes(size, label)?;
        Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        }))
    }
}

/// Find a usable adapter under `options`.
async fn select_adapter(options: &ContextOptions) -> Result<wgpu::Adapter, RenderError> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: options.backends,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        backend_options: wgpu::BackendOptions::default(),
        display: None,
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: options.power_preference,
            force_fallback_adapter: options.force_fallback_adapter,
            compatible_surface: None,
            apply_limit_buckets: false,
        })
        .await
        .map_err(|_| RenderError::NoAdapter {
            required_fallback: options.force_fallback_adapter,
        })?;

    let info = adapter.get_info();
    adapter_is_acceptable(info.device_type, options)?;
    check_required_features(options.required_features, adapter.features())?;
    check_required_limits(&options.effective_limits(), &adapter.limits())?;
    Ok(adapter)
}

/// Discover what would be used, without creating a device.
///
/// Useful for reporting backend status without paying for device creation.
///
/// # Errors
/// As [`request_context`].
pub async fn probe_adapter(options: &ContextOptions) -> Result<BackendInfo, RenderError> {
    let adapter = select_adapter(options).await?;
    Ok(info_from(&adapter))
}

/// Create a GPU context.
///
/// Asynchronous so that a UI thread never blocks on adapter discovery or device
/// creation.
///
/// # Errors
/// [`RenderError::NoAdapter`], [`RenderError::UnsupportedCapability`],
/// [`RenderError::ResourceLimit`], or [`RenderError::DeviceCreationFailed`].
pub async fn request_context(options: ContextOptions) -> Result<GpuContext, RenderError> {
    let adapter = select_adapter(&options).await?;
    let limits = options.effective_limits();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some(&options.label),
            required_features: options.required_features,
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|e| RenderError::DeviceCreationFailed {
            message: format!("{e:?}"),
        })?;

    let mut info = info_from(&adapter);
    // Report the limits actually in force on the device, which may be lower than
    // the adapter's when requirements were tightened.
    let device_limits = device.limits().clone();
    info.max_buffer_size = device_limits.max_buffer_size;
    info.max_storage_buffer_binding_size = device_limits.max_storage_buffer_binding_size;
    info.max_texture_dimension_2d = device_limits.max_texture_dimension_2d;
    info.max_texture_dimension_3d = device_limits.max_texture_dimension_3d;

    Ok(GpuContext {
        device: Arc::new(device),
        queue: Arc::new(queue),
        info,
        limits: device_limits,
    })
}

/// Blocking wrapper around [`request_context`].
///
/// **Blocks the calling thread** while the GPU is initialised. Call it from a
/// worker thread, not from a UI event loop; that is the reason the asynchronous
/// entry point exists.
///
/// # Errors
/// As [`request_context`].
pub fn init_blocking(options: ContextOptions) -> Result<GpuContext, RenderError> {
    pollster::block_on(request_context(options))
}

/// Bytes per row required by `wgpu` for texture and buffer copies.
pub const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

/// Round `value` up to the copy row alignment.
pub fn align_row_bytes(value: u32) -> u32 {
    let a = COPY_BYTES_PER_ROW_ALIGNMENT;
    value.div_ceil(a) * a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_are_permissive_and_typed() {
        let o = ContextOptions::new();
        assert!(o.required_features.is_empty());
        assert!(o.required_limits.is_none());
        assert!(o.allow_software_fallback);
        assert!(!o.force_fallback_adapter);
        assert_eq!(
            o.effective_limits().max_buffer_size,
            wgpu::Limits::default().max_buffer_size
        );
    }

    #[test]
    fn options_builders_are_chainable() {
        let o = ContextOptions::new()
            .with_backends(wgpu::Backends::METAL)
            .with_label("test")
            .without_software_fallback()
            .with_required_features(wgpu::Features::empty());
        assert_eq!(o.label, "test");
        assert_eq!(o.backends, wgpu::Backends::METAL);
        assert!(!o.allow_software_fallback);
    }

    #[test]
    fn software_adapters_are_rejected_only_when_disallowed() {
        let permissive = ContextOptions::new();
        assert!(adapter_is_acceptable(wgpu::DeviceType::Cpu, &permissive).is_ok());
        assert!(adapter_is_acceptable(wgpu::DeviceType::IntegratedGpu, &permissive).is_ok());

        let strict = ContextOptions::new().without_software_fallback();
        match adapter_is_acceptable(wgpu::DeviceType::Cpu, &strict) {
            Err(RenderError::NoAdapter { required_fallback }) => assert!(!required_fallback),
            other => panic!("expected NoAdapter, got {other:?}"),
        }
        assert!(adapter_is_acceptable(wgpu::DeviceType::DiscreteGpu, &strict).is_ok());
    }

    #[test]
    fn missing_features_are_reported_by_name() {
        let available = wgpu::Features::empty();
        check_required_features(wgpu::Features::empty(), available).unwrap();

        let err = check_required_features(
            wgpu::Features::DEPTH_CLIP_CONTROL,
            wgpu::Features::DEPTH_CLIP_CONTROL,
        );
        assert!(err.is_ok(), "a feature the adapter has must be accepted");

        let err =
            check_required_features(wgpu::Features::DEPTH_CLIP_CONTROL, available).unwrap_err();
        assert!(err.is_capability_mismatch());
        let msg = err.to_string();
        assert!(msg.contains("DEPTH_CLIP_CONTROL"), "{msg}");
    }

    #[test]
    fn limits_are_compared_field_by_field() {
        let available = wgpu::Limits::default();
        check_required_limits(&available, &available).unwrap();

        let too_big = wgpu::Limits {
            max_buffer_size: available.max_buffer_size + 1,
            ..Default::default()
        };
        match check_required_limits(&too_big, &available) {
            Err(RenderError::ResourceLimit {
                limit, requested, ..
            }) => {
                assert!(limit.contains("max_buffer_size"));
                assert_eq!(requested, available.max_buffer_size + 1);
            }
            other => panic!("expected ResourceLimit, got {other:?}"),
        }

        let small = wgpu::Limits {
            max_buffer_size: available.max_buffer_size / 2,
            ..Default::default()
        };
        check_required_limits(&small, &available).unwrap();

        let bad_tex = wgpu::Limits {
            max_texture_dimension_2d: available.max_texture_dimension_2d + 4096,
            ..Default::default()
        };
        assert!(check_required_limits(&bad_tex, &available).is_err());

        let bad_binding = wgpu::Limits {
            max_storage_buffer_binding_size: available.max_storage_buffer_binding_size + 1,
            ..Default::default()
        };
        match check_required_limits(&bad_binding, &available) {
            Err(RenderError::ResourceLimit { limit, .. }) => {
                assert!(limit.contains("max_storage_buffer_binding_size"))
            }
            other => panic!("expected ResourceLimit, got {other:?}"),
        }
    }

    #[test]
    fn row_alignment_is_to_256_bytes() {
        assert_eq!(align_row_bytes(4), 256);
        assert_eq!(align_row_bytes(256), 256);
        assert_eq!(align_row_bytes(257), 512);
        // a 141-wide RGBA row is 564 bytes -> 768
        assert_eq!(align_row_bytes(141 * 4), 768);
    }

    #[test]
    fn context_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GpuContext>();
        assert_send_sync::<BackendInfo>();
    }

    #[test]
    fn backend_info_reports_readably() {
        let info = BackendInfo {
            backend: "metal".into(),
            adapter_name: "Apple M4 Max".into(),
            device_type: "integrated-gpu".into(),
            driver: "Metal".into(),
            driver_info: "4.0".into(),
            vendor_id: 0,
            device_id: 0,
            max_texture_dimension_2d: 16384,
            max_texture_dimension_3d: 2048,
            max_buffer_size: 1 << 30,
            max_storage_buffer_binding_size: 1 << 27,
        };
        assert!(!info.is_software());
        assert_eq!(info.summary(), "metal / Apple M4 Max (integrated-gpu)");
        assert!(info.report().contains("16384/2048"));
    }

    /// A context on a machine with no GPU must be an `Err`, never a panic.
    #[test]
    fn no_adapter_is_an_error_not_a_panic() {
        let options = ContextOptions::new()
            .with_backends(wgpu::Backends::empty())
            .with_label("atoms-render-capability-probe");
        match init_blocking(options) {
            Err(e) => assert!(
                e.is_unavailable() || e.is_capability_mismatch(),
                "unexpected error class: {e}"
            ),
            Ok(ctx) => {
                // Some backends treat an empty `Backends` mask as "all"; if a
                // context came back, it must still be internally consistent.
                assert!(!ctx.info().backend.is_empty());
            }
        }
    }
}
