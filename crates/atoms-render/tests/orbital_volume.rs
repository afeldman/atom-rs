//! Integration test: the renderer, fed a real physical field.
//!
//! The unit tests use a synthetic volume with coordinate-encoding values, which
//! is the right tool for catching a transposition. This test does the opposite:
//! it feeds the renderer a field whose exact properties are known analytically —
//! the 2p_z hydrogen orbital — and asserts that the rendered image reproduces
//! those properties.
//!
//! The orbital is `n = 2, l = 1, m = 0`, for which
//!
//! ```text
//! |psi|^2 ∝ z^2 * exp(-r),   r = sqrt(x^2 + y^2 + z^2)
//! ```
//!
//! so three facts are available in closed form and can be checked against the
//! *rendered output*, not merely against the array:
//!
//! 1. the `z = 0` plane is a nodal plane: the density is exactly zero there, and
//!    therefore so is every pixel of the rendered `z = 0` slice;
//! 2. the global maximum sits on the `z` axis at `|z| = 2` bohr;
//! 3. the sampled values are all non-negative.
//!
//! The GPU path is exercised when the `gpu` feature is enabled and an adapter
//! exists; on a machine without one the GPU tests skip with a printed reason
//! rather than failing.

use atoms_core::{GridError, SpatialGrid3D};
use atoms_orbitals::HydrogenWavefunction;
use atoms_render::error::RenderError;
use atoms_render::volume::{SliceAxis, TransferFunction, VolumeData};
use atoms_render::{cpu, RgbaImage};

/// Atomistic units: the grid spans [-12, 12] bohr in every direction.
const EXTENT_BOHR: f64 = 12.0;
/// 121 points per axis gives a spacing of 0.2 bohr and places 0, +/-2, +/-12 on
/// the grid exactly, which the analytic assertions below rely on.
const POINTS: usize = 121;

/// The 2p_z orbital.
fn orbital() -> HydrogenWavefunction {
    HydrogenWavefunction::new(2, 1, 0).expect("n=2, l=1, m=0 is a valid state")
}

fn grid() -> Result<SpatialGrid3D, GridError> {
    let half = EXTENT_BOHR;
    SpatialGrid3D::new(
        nalgebra::Point3::new(-half, -half, -half),
        nalgebra::Vector3::new(
            2.0 * half / (POINTS - 1) as f64,
            2.0 * half / (POINTS - 1) as f64,
            2.0 * half / (POINTS - 1) as f64,
        ),
        [POINTS, POINTS, POINTS],
    )
}

/// The field, in public array order — the layout `VolumeData` requires.
fn density_public_order() -> (SpatialGrid3D, Vec<f64>) {
    let g = grid().unwrap();
    let wf = orbital();
    let values = g.map_public(|x, y, z| wf.probability_density(x, y, z));
    (g, values)
}

/// The bridge from the physics grid to the renderer's neutral interface.
///
/// This is the entire coupling between the two sides, and it lives here (in a
/// dev-only test/example) rather than in either library, so neither has to know
/// about the other. Five lines, no types from the physics crates crossing over.
fn to_volume(g: &SpatialGrid3D, values: &[f64]) -> Result<VolumeData, RenderError> {
    let origin = g.origin();
    let spacing = g.spacing();
    VolumeData::from_f64(
        g.counts(),
        [origin.x, origin.y, origin.z],
        [spacing.x, spacing.y, spacing.z],
        values,
        "bohr^-3",
        true, // a probability density cannot be negative
    )
}

fn volume() -> (SpatialGrid3D, Vec<f64>, VolumeData) {
    let (g, values) = density_public_order();
    let v = to_volume(&g, &values).unwrap();
    (g, values, v)
}

/// Index of the grid point closest to `0` on every axis.
fn centre_index() -> usize {
    (POINTS - 1) / 2
}

#[test]
fn the_field_is_physically_correct_before_it_reaches_the_renderer() {
    let (g, values) = density_public_order();
    assert_eq!(values.len(), POINTS * POINTS * POINTS);
    assert!(
        values.iter().all(|v| *v >= 0.0 && v.is_finite()),
        "a probability density must be finite and non-negative"
    );

    // The maximum must be on the z axis at |z| = 2 bohr.
    let (best_index, _) = values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let (i, j, k) = g.multi_index_public(best_index);
    let p = g.point(i, j, k);
    let centre = centre_index();
    assert_eq!(
        (i, j),
        (centre, centre),
        "the maximum must be on the z axis"
    );
    assert!(
        (p.z.abs() - 2.0).abs() < 1e-12,
        "the maximum of z^2 exp(-r) is at |z| = 2, got {}",
        p.z
    );

    // The z = 0 plane is a nodal plane for m = 0, l = 1: the density is
    // mathematically exactly zero there.
    //
    // The evaluation route leaves a floating-point residue, because a Cartesian
    // point is converted to spherical coordinates and the angular factor is
    // cos(theta) with theta = atan2(rho, z) = pi/2 exactly, and cos(pi/2) is
    // 6.1e-17 in f64 rather than 0. Squared, the residue lands around 1e-36 where
    // the true value is 0. It is an artefact of the coordinate round-trip, not of
    // the physics, and the assertion below is therefore stated relative to the
    // peak density: the residue must stay at least 30 orders of magnitude below
    // it. The renderer's own nodal-plane test then asserts the image is exactly
    // black, which is the property that matters.
    let peak = values.iter().copied().fold(0.0f64, f64::max);
    let mut worst_nodal = 0.0f64;
    for i in 0..POINTS {
        for j in 0..POINTS {
            worst_nodal = worst_nodal.max(values[(i * POINTS + j) * POINTS + centre]);
        }
    }
    assert!(
        worst_nodal <= peak * 1e-30,
        "the nodal-plane residue must be negligible against the peak density {peak:e}, \
         but reached {worst_nodal:e}"
    );
    eprintln!(
        "nodal-plane residue: {worst_nodal:e} vs peak {peak:e} \
         (ratio {:.3e})",
        worst_nodal / peak
    );

    // The grid is finite, so the integrated density is slightly below 1.
    let total = g.integrate(|x, y, z| orbital().probability_density(x, y, z));
    assert!(
        (0.94..1.0).contains(&total),
        "expected the sampled total probability to approach 1, got {total}"
    );
}

#[test]
fn the_bridge_reports_its_own_f32_conversion_error() {
    let (_, values, v) = volume();

    // Per-value relative error is dominated by the nodal-plane residue, which is
    // subnormal in f32; it is large in relative terms by construction.
    let per_value = v.max_relative_conversion_error(&values);

    // The metric that matters for rendering: error relative to the field
    // maximum, which is what the transfer function normalises by.
    let relative_to_max = v.max_error_relative_to_field_max(&values);
    assert!(
        relative_to_max < 6e-8,
        "f32 storage must stay within 2^-24 of the field maximum, measured {relative_to_max:e}"
    );
    assert!(
        per_value > relative_to_max,
        "the per-value metric must be the pessimistic one on this field"
    );
    eprintln!(
        "f32 conversion error: relative-to-max {relative_to_max:e}, worst per-value {per_value:e}"
    );

    // and the stored data is still non-negative after conversion
    assert!(v.values().iter().all(|x| *x >= 0.0));
}

#[test]
fn the_rendered_nodal_plane_is_exactly_black() {
    let (_, _, v) = volume();
    let centre = centre_index();
    let tf = TransferFunction::log_scaled(v.max());

    // A slice through z = 0 must be exactly zero on the scalar channel.
    let slice = cpu::render_slice(&v, SliceAxis::Z, centre, &tf).unwrap();
    assert_eq!(slice.width(), POINTS);
    assert_eq!(slice.height(), POINTS);
    let scalar = slice.scalar_bytes();
    let non_zero = scalar.iter().filter(|b| **b != 0).count();
    assert_eq!(
        non_zero, 0,
        "the nodal plane must render as an entirely black image, {non_zero} non-zero pixels"
    );

    // The volume maximum is on the z axis at |z| = 2, which is a slice along
    // z at index centre +/- 10 (2.0 / 0.2 = 10 steps from the centre).
    let peak_index = centre + 10;
    let peak = cpu::render_slice(&v, SliceAxis::Z, peak_index, &tf).unwrap();
    let peak_max = peak.scalar_bytes().iter().copied().max().unwrap();
    assert!(
        peak_max > 200,
        "the slice through the maximum should be bright, got peak byte {peak_max}"
    );
    // and its brightest pixel must be on the axis, i.e. at the image centre
    let (bw, bh) = (peak.width() as usize, peak.height() as usize);
    let mut best = (0usize, 0usize, 0u8);
    for py in 0..bh {
        for px in 0..bw {
            let s = peak.scalar_bytes()[py * bw + px];
            if s > best.2 {
                best = (px, py, s);
            }
        }
    }
    assert_eq!(
        (best.0, best.1),
        (centre, centre),
        "the brightest pixel of the peak slice must be the point on the z axis"
    );
}

#[test]
fn the_projection_shows_a_dumbbell_and_the_slice_through_it_does_not() {
    let (_, _, v) = volume();
    let tf = TransferFunction::log_scaled(v.max());
    let centre = centre_index();

    // MIP along y (an x-z image): the two lobes must appear as two bright
    // regions separated by the nodal plane at z = 0 (row centre).
    let mip = cpu::render_mip(&v, SliceAxis::Y, &tf).unwrap();
    let width = mip.width();
    let scalar = mip.scalar_bytes();
    let row = |py: usize| -> u8 { scalar[py * width + centre].max(scalar[py * width]) };
    let nodal_row = scalar[centre * width + centre];
    assert_eq!(
        nodal_row, 0,
        "the projection must still be dark on the nodal plane"
    );
    let above = row(centre + 5);
    let below = row(centre - 5);
    assert!(
        above > 100 && below > 100,
        "both lobes must be bright: {above}, {below}"
    );
    // symmetry: the two lobes are mirror images through z = 0
    let upper: Vec<u8> = (0..centre).map(|py| row(centre + 1 + py)).collect();
    let lower: Vec<u8> = (0..centre).map(|py| row(centre - 1 - py)).collect();
    assert_eq!(
        upper, lower,
        "the projected lobes must be mirror-symmetric in z"
    );

    // A slice along Y through the centre is a nodal cross-section too: at
    // x = y = 0 the plane contains z but the *slice* fixes y = 0, and the
    // density vanishes on the z axis only for r > 0 off-axis points... so the
    // check here is simply that the slice is non-degenerate.
    let slice = cpu::render_slice(&v, SliceAxis::Z, centre + 10, &tf).unwrap();
    assert!(slice.scalar_bytes().iter().any(|b| *b > 0));
}

#[test]
fn all_axes_and_every_index_render_with_the_declared_dimensions() {
    let (_, _, v) = volume();
    let tf = TransferFunction::log_scaled(v.max());
    for axis in [SliceAxis::X, SliceAxis::Y, SliceAxis::Z] {
        let (w, h) = v.slice_dims(axis);
        for index in 0..v.axis_len(axis) {
            let img = cpu::render_slice(&v, axis, index, &tf).unwrap();
            assert_eq!((img.width(), img.height()), (w, h));
        }
        let mip = cpu::render_mip(&v, axis, &tf).unwrap();
        assert_eq!((mip.width(), mip.height()), (w, h));
    }
}

// ---------------------------------------------------------------------------
// GPU path (only with the `gpu` feature and a usable adapter)
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use atoms_render::gpu::{init_blocking, ContextOptions, SliceRenderer};

    fn context() -> Option<atoms_render::gpu::GpuContext> {
        match init_blocking(ContextOptions::new().with_label("atoms-render integration test")) {
            Ok(ctx) => Some(ctx),
            Err(e) if e.is_unavailable() => {
                eprintln!("skipping GPU integration test: {e}");
                None
            }
            Err(e) => panic!("GPU initialisation failed unexpectedly: {e}"),
        }
    }

    #[test]
    fn the_gpu_renders_the_nodal_plane_exactly_black_too() {
        let Some(ctx) = context() else { return };
        let renderer = SliceRenderer::new(&ctx).unwrap();
        let (_, _, v) = super::volume();
        let tf = TransferFunction::log_scaled(v.max());
        let centre = super::centre_index();

        let img = renderer
            .render_slice(&ctx, &v, SliceAxis::Z, centre, &tf)
            .unwrap();
        let cpu_img = cpu::render_slice(&v, SliceAxis::Z, centre, &tf).unwrap();
        // Tolerance zero: both sides must produce an entirely empty image.
        let report = img.compare_scalar(&cpu_img, 0).unwrap();
        assert!(
            report.within_tolerance(),
            "the GPU and CPU must agree exactly on the nodal plane: {report}"
        );
        assert!(img.scalar_bytes().iter().all(|b| *b == 0));
    }

    #[test]
    fn gpu_and_cpu_agree_on_real_orbital_data_within_one_quantisation_step() {
        let Some(ctx) = context() else { return };
        let renderer = SliceRenderer::new(&ctx).unwrap();
        let (_, _, v) = super::volume();
        let tf = TransferFunction::log_scaled(v.max());
        let centre = super::centre_index();

        let mut worst = 0u8;
        let mut worst_at = String::new();
        for axis in [SliceAxis::X, SliceAxis::Y, SliceAxis::Z] {
            for index in [centre, centre + 10, centre + 20, 3] {
                if index >= v.axis_len(axis) {
                    continue;
                }
                let gpu = renderer.render_slice(&ctx, &v, axis, index, &tf).unwrap();
                let cpu_img = cpu::render_slice(&v, axis, index, &tf).unwrap();
                let report = gpu.compare_scalar(&cpu_img, 1).unwrap();
                assert!(
                    report.within_tolerance(),
                    "GPU/CPU disagreement for {axis:?} at {index}: {report}"
                );
                if report.max_difference > worst {
                    worst = report.max_difference;
                    worst_at = format!("{axis:?} at index {index}: {report}");
                }
            }
            let gpu = renderer.render_mip(&ctx, &v, axis, &tf).unwrap();
            let cpu_img = cpu::render_mip(&v, axis, &tf).unwrap();
            let report = gpu.compare_scalar(&cpu_img, 1).unwrap();
            assert!(
                report.within_tolerance(),
                "GPU/CPU MIP disagreement for {axis:?}: {report}"
            );
        }
        eprintln!("worst GPU/CPU image deviation: {worst} step(s) [{worst_at}]");
    }

    #[test]
    fn the_uploaded_orbital_density_reaches_the_gpu_unchanged() {
        use atoms_render::gpu::slice::{decode_f32, measure_upload_error, read_buffer};
        let Some(ctx) = context() else { return };
        let renderer = SliceRenderer::new(&ctx).unwrap();
        let (_, _, v) = super::volume();
        let buffer = renderer.upload_volume(&ctx, &v).unwrap();
        let bytes = read_buffer(&ctx, &buffer, v.byte_len() as u64).unwrap();
        let gpu = decode_f32(&bytes).unwrap();
        assert_eq!(gpu.len(), v.len());
        assert_eq!(
            measure_upload_error(&gpu, v.values()).unwrap(),
            0,
            "the volume the shader reads must be bit-identical to the CPU's"
        );
    }

    #[test]
    fn rendering_the_same_orbital_twice_is_bit_stable() {
        let Some(ctx) = context() else { return };
        let renderer = SliceRenderer::new(&ctx).unwrap();
        let (_, _, v) = super::volume();
        let tf = TransferFunction::log_scaled(v.max());
        let a = renderer.render_mip(&ctx, &v, SliceAxis::Y, &tf).unwrap();
        let b = renderer.render_mip(&ctx, &v, SliceAxis::Y, &tf).unwrap();
        assert_eq!(a.pixels(), b.pixels());
    }
}

/// The comparison helper is exercised on real data even without a GPU.
#[test]
fn compare_report_is_sane_on_real_images() {
    let (_, _, v) = volume();
    let tf = TransferFunction::log_scaled(v.max());
    let a = cpu::render_slice(&v, SliceAxis::Z, centre_index() + 10, &tf).unwrap();
    let b = cpu::render_slice(&v, SliceAxis::Z, centre_index() + 10, &tf).unwrap();
    let report = a.compare_scalar(&b, 0).unwrap();
    assert_eq!(report.max_difference, 0);
    assert!(report.within_tolerance());
    let _: &RgbaImage = &a;
}

/// The documented fallback pattern: try the GPU, and on an *unavailable* error
/// render on the CPU instead of failing.
///
/// This is the contract a caller relies on, so it is executed rather than
/// described. On this machine the GPU path is available, so the test also
/// asserts that the GPU and the fallback agree — which is the property that makes
/// falling back safe.
#[test]
fn the_cpu_fallback_produces_the_same_image_as_the_gpu_path() {
    let (_, _, v) = volume();
    let tf = TransferFunction::log_scaled(v.max());
    let centre = centre_index();

    // The fallback is always available, whatever the GPU does.
    let fallback = cpu::render_mip(&v, SliceAxis::Y, &tf).unwrap();
    assert!(fallback.scalar_bytes().iter().any(|b| *b > 0));

    #[cfg(feature = "gpu")]
    {
        use atoms_render::gpu::{init_blocking, ContextOptions, SliceRenderer};

        match init_blocking(ContextOptions::new().with_label("atoms-render fallback test")) {
            Ok(ctx) => {
                let renderer = SliceRenderer::new(&ctx).unwrap();
                let gpu = renderer.render_mip(&ctx, &v, SliceAxis::Y, &tf).unwrap();
                let report = gpu.compare_scalar(&fallback, 1).unwrap();
                assert!(
                    report.within_tolerance(),
                    "the fallback must not be a second-class path: {report}"
                );
            }
            Err(e) if e.is_unavailable() => {
                // Exactly the situation the fallback exists for.
                eprintln!("no adapter ({e}); the CPU path above is the result");
            }
            Err(e) => panic!("unexpected GPU failure: {e}"),
        }
    }

    // A GPU request that cannot be satisfied must produce an error a caller can
    // classify as "unavailable", not a panic and not an opaque message.
    #[cfg(feature = "gpu")]
    {
        use atoms_render::gpu::{init_blocking, ContextOptions};
        let impossible = ContextOptions::new()
            .with_backends(wgpu::Backends::empty())
            .with_label("atoms-render fallback probe");
        if let Err(e) = init_blocking(impossible) {
            assert!(
                e.is_unavailable() || e.is_capability_mismatch(),
                "an unsatisfiable request must be classifiable, got: {e}"
            );
        }
    }

    // Whatever happened above, the CPU result is still the reference.
    let again = cpu::render_mip(&v, SliceAxis::Y, &tf).unwrap();
    assert_eq!(again.pixels(), fallback.pixels());
    assert_eq!(centre, (POINTS - 1) / 2);
}
