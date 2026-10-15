//! Deterministic render of a real orbital, written to disk for inspection.
//!
//! This is the M2-B acceptance artefact: it renders a `2p_z` hydrogen orbital
//! through the portable `wgpu` path and through the CPU reference, compares them
//! numerically, and writes PNGs plus a metadata record.
//!
//! ```text
//! cargo run -p atoms-render --features gpu --example orbital_slice -- [outdir]
//! ```
//!
//! Everything it prints is measured during the run: no figure in the output is an
//! estimate. Images are written even when no GPU is present — the CPU path is
//! always available, and the metadata records which backend produced which file.

use std::io;
use std::path::{Path, PathBuf};

use atoms_core::{GridError, SpatialGrid3D};
use atoms_orbitals::HydrogenWavefunction;
use atoms_render::error::RenderError;
use atoms_render::volume::{SliceAxis, TransferFunction, VolumeData};
use atoms_render::{cpu, RgbaImage};

const EXTENT_BOHR: f64 = 12.0;
const POINTS: usize = 141;

fn grid() -> Result<SpatialGrid3D, GridError> {
    let step = 2.0 * EXTENT_BOHR / (POINTS - 1) as f64;
    SpatialGrid3D::new(
        nalgebra::Point3::new(-EXTENT_BOHR, -EXTENT_BOHR, -EXTENT_BOHR),
        nalgebra::Vector3::new(step, step, step),
        [POINTS, POINTS, POINTS],
    )
}

/// The one documented bridge from a physics grid to the renderer.
fn to_volume(g: &SpatialGrid3D, values: &[f64]) -> Result<VolumeData, RenderError> {
    let o = g.origin();
    let s = g.spacing();
    VolumeData::from_f64(
        g.counts(),
        [o.x, o.y, o.z],
        [s.x, s.y, s.z],
        values,
        "bohr^-3",
        true,
    )
}

type Manifest = Vec<(String, String, u64, String)>;

fn emit(
    outdir: &Path,
    manifest: &mut Manifest,
    name: &str,
    backend: &str,
    img: &RgbaImage,
    note: &str,
) -> io::Result<()> {
    let file = outdir.join(name);
    img.write_png(&file)?;
    let bytes = std::fs::metadata(&file)?.len();
    manifest.push((
        name.to_string(),
        backend.to_string(),
        bytes,
        note.to_string(),
    ));
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let outdir: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/orbital-artifacts".to_string())
        .into();
    std::fs::create_dir_all(&outdir)?;

    // ---- the physical field -------------------------------------------------
    let t_build = std::time::Instant::now();
    let g = grid()?;
    let wf = HydrogenWavefunction::new(2, 1, 0)?; // 2p_z
    let values = g.map_public(|x, y, z| wf.probability_density(x, y, z));
    let build_time = t_build.elapsed();

    let t_convert = std::time::Instant::now();
    let volume = to_volume(&g, &values)?;
    let convert_time = t_convert.elapsed();

    let centre = (POINTS - 1) / 2;
    let step = 2.0 * EXTENT_BOHR / (POINTS - 1) as f64;
    let peak_index = centre + (2.0 / step).round() as usize; // z = +2 bohr
    let tf = TransferFunction::log_scaled(volume.max());
    let total: f64 = values.iter().sum::<f64>() * step.powi(3);

    println!("== atoms-render / M2-B deterministic scene ==");
    println!("orbital                 2p_z (n=2, l=1, m=0), hydrogen, atomic units");
    println!(
        "grid                    {POINTS}^3 over [-{EXTENT_BOHR}, {EXTENT_BOHR}] bohr, spacing {step:.6} bohr"
    );
    println!("points                  {}", volume.len());
    println!("volume bytes (f32)      {}", volume.byte_len());
    println!(
        "f32 error rel. to max   {:.3e}  (bound 6e-8 = 2^-24)",
        volume.max_error_relative_to_field_max(&values)
    );
    println!(
        "f32 error worst per-val {:.3e}  (subnormal at the nodal plane)",
        volume.max_relative_conversion_error(&values)
    );
    println!("max density             {:.6e} bohr^-3", volume.max());
    println!("integrated probability  {total:.6}");
    println!("field build             {build_time:?}");
    println!("f64 -> VolumeData       {convert_time:?}");

    // ---- CPU reference: always available ------------------------------------
    let mut manifest: Manifest = Vec::new();

    let t_cpu = std::time::Instant::now();
    let cpu_slice = cpu::render_slice(&volume, SliceAxis::Z, peak_index, &tf)?;
    let cpu_nodal = cpu::render_slice(&volume, SliceAxis::Z, centre, &tf)?;
    let cpu_mip = cpu::render_mip(&volume, SliceAxis::Y, &tf)?;
    let cpu_time = t_cpu.elapsed();

    emit(
        &outdir,
        &mut manifest,
        "cpu_mip_y.png",
        "cpu",
        &cpu_mip,
        "maximum-intensity projection along y: the two 2p_z lobes",
    )?;
    emit(
        &outdir,
        &mut manifest,
        "cpu_slice_z_plus2.png",
        "cpu",
        &cpu_slice,
        "slice at z = +2 bohr, through the density maximum",
    )?;
    emit(
        &outdir,
        &mut manifest,
        "cpu_slice_nodal_plane.png",
        "cpu",
        &cpu_nodal,
        "slice at z = 0, the nodal plane: must be entirely black",
    )?;

    println!("\n== CPU reference ==");
    println!("three images             {cpu_time:?}");
    let nodal_nonzero = cpu_nodal.scalar_bytes().iter().filter(|b| **b != 0).count();
    println!(
        "nodal plane              {nodal_nonzero} of {} pixels non-zero (expected 0)",
        cpu_nodal.scalar_bytes().len()
    );
    let peak_max = cpu_slice.scalar_bytes().iter().copied().max().unwrap();
    println!("peak slice max byte      {peak_max} of 255");

    // ---- GPU path: only if an adapter exists --------------------------------
    let backend_report: String = {
        #[cfg(feature = "gpu")]
        {
            use atoms_render::gpu::{init_blocking, ContextOptions, SliceRenderer};

            match init_blocking(ContextOptions::new().with_label("atoms-render example")) {
                Ok(ctx) => {
                    println!("\n== GPU ==");
                    let report = ctx.info().report();
                    print!("{report}");

                    let renderer = SliceRenderer::new(&ctx)?;

                    // Upload cost, measured separately from render cost.
                    let t_up = std::time::Instant::now();
                    let buffer = renderer.upload_volume(&ctx, &volume)?;
                    let upload_time = t_up.elapsed();

                    let t_gpu = std::time::Instant::now();
                    let gpu_slice = renderer.render_with_buffer(
                        &ctx,
                        &volume,
                        &buffer,
                        SliceAxis::Z,
                        0,
                        peak_index,
                        &tf,
                    )?;
                    let gpu_nodal = renderer.render_with_buffer(
                        &ctx,
                        &volume,
                        &buffer,
                        SliceAxis::Z,
                        0,
                        centre,
                        &tf,
                    )?;
                    let gpu_mip = renderer.render_with_buffer(
                        &ctx,
                        &volume,
                        &buffer,
                        SliceAxis::Y,
                        1,
                        0,
                        &tf,
                    )?;
                    let gpu_time = t_gpu.elapsed();

                    let backend = ctx.info().backend.clone();
                    emit(
                        &outdir,
                        &mut manifest,
                        "gpu_mip_y.png",
                        &backend,
                        &gpu_mip,
                        "maximum-intensity projection along y, rendered on the GPU",
                    )?;
                    emit(
                        &outdir,
                        &mut manifest,
                        "gpu_slice_z_plus2.png",
                        &backend,
                        &gpu_slice,
                        "slice at z = +2 bohr, rendered on the GPU",
                    )?;
                    emit(
                        &outdir,
                        &mut manifest,
                        "gpu_slice_nodal_plane.png",
                        &backend,
                        &gpu_nodal,
                        "nodal plane, rendered on the GPU",
                    )?;

                    println!(
                        "upload                   {upload_time:?}  ({} bytes)",
                        volume.byte_len()
                    );
                    println!("three images             {gpu_time:?}");

                    let comparisons = [
                        ("slice z=+2", &gpu_slice, &cpu_slice),
                        ("nodal plane z=0", &gpu_nodal, &cpu_nodal),
                        ("MIP along y", &gpu_mip, &cpu_mip),
                    ];
                    let mut report_md = String::from(
                        "# M2-B GPU/CPU agreement\n\n\
                         The scalar (alpha) channel of the GPU image is compared against the CPU\n\
                         reference. Deviations are in quantisation steps out of 255.\n\n\
                         | image | pixels | max abs d | mean abs d | differing | beyond tol 1 |\n\
                         |---|---|---|---|---|---|\n",
                    );
                    println!("\n== GPU vs CPU (scalar channel) ==");
                    for (name, gpu_img, cpu_img) in comparisons {
                        let r = gpu_img.compare_scalar(cpu_img, 1)?;
                        println!("{name:20} {r}");
                        report_md.push_str(&format!(
                            "| {name} | {} | {} | {:.5} | {} | {} |\n",
                            r.pixels,
                            r.max_difference,
                            r.mean_difference,
                            r.differing,
                            r.beyond_tolerance
                        ));
                    }
                    std::fs::write(outdir.join("gpu_cpu_agreement.md"), report_md)?;

                    if let Ok(bytes) = atoms_render::gpu::slice::read_buffer(
                        &ctx,
                        &buffer,
                        volume.byte_len() as u64,
                    ) {
                        if let Ok(gpu_values) = atoms_render::gpu::slice::decode_f32(&bytes) {
                            match atoms_render::gpu::slice::measure_upload_error(
                                &gpu_values,
                                volume.values(),
                            ) {
                                Ok(0) => {
                                    println!("upload integrity         bit-identical (max ULP = 0)")
                                }
                                Ok(ulp) => {
                                    println!("upload integrity         max ULP difference = {ulp}")
                                }
                                Err(e) => println!("upload integrity         not measured: {e}"),
                            }
                        }
                    }

                    report
                }
                Err(e) if e.is_unavailable() => {
                    println!("\n== GPU ==\nno adapter: {e}");
                    format!("no GPU adapter available: {e}\n")
                }
                Err(e) => return Err(Box::new(e)),
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            println!("\n== GPU ==\nbuilt without the `gpu` feature; CPU path only");
            "built without the `gpu` feature\n".to_string()
        }
    };

    // ---- metadata -----------------------------------------------------------
    let mut meta = String::new();
    meta.push_str("{\n");
    meta.push_str("  \"scene\": \"hydrogen 2p_z probability density\",\n");
    meta.push_str("  \"quantum_numbers\": {\"n\": 2, \"l\": 1, \"m\": 0},\n");
    meta.push_str(
        "  \"units\": {\"length\": \"bohr\", \"density\": \"bohr^-3\", \"energy\": \"hartree\"},\n",
    );
    meta.push_str(&format!(
        "  \"grid\": {{\"points\": {POINTS}, \"extent_bohr\": {EXTENT_BOHR}, \"spacing_bohr\": {step:.10}}},\n"
    ));
    meta.push_str(&format!(
        "  \"origin_bohr\": [{}, {}, {}],\n",
        -EXTENT_BOHR, -EXTENT_BOHR, -EXTENT_BOHR
    ));
    meta.push_str("  \"array_order\": \"public: index(i,j,k) = (i*ny + j)*nz + k, k fastest\",\n");
    meta.push_str(&format!("  \"max_density\": {:.10e},\n", volume.max()));
    meta.push_str(&format!("  \"integrated_probability\": {total:.10},\n"));
    meta.push_str(&format!(
        "  \"max_relative_f32_error\": {:.6e},\n",
        volume.max_relative_conversion_error(&values)
    ));
    meta.push_str(&format!(
        "  \"transfer_function\": {{\"vmax\": {:.10e}, \"compression\": 20.0}},\n",
        tf.vmax
    ));
    meta.push_str(&format!(
        "  \"timings_s\": {{\"field_build\": {:.6}, \"convert\": {:.6}, \"cpu_three_images\": {:.6}}},\n",
        build_time.as_secs_f64(),
        convert_time.as_secs_f64(),
        cpu_time.as_secs_f64()
    ));
    meta.push_str("  \"backend\": \"");
    meta.push_str(&backend_report.replace('"', "'").replace('\n', " | "));
    meta.push_str("\",\n");
    meta.push_str("  \"images\": [\n");
    for (index, (name, backend, bytes, note)) in manifest.iter().enumerate() {
        let comma = if index + 1 == manifest.len() { "" } else { "," };
        meta.push_str(&format!(
            "    {{\"file\": \"{name}\", \"backend\": \"{backend}\", \"bytes\": {bytes}, \"note\": \"{note}\"}}{comma}\n"
        ));
    }
    meta.push_str("  ],\n");
    meta.push_str("  \"software_version\": \"");
    meta.push_str(env!("CARGO_PKG_VERSION"));
    meta.push_str("\"\n}\n");
    std::fs::write(outdir.join("scene.json"), &meta)?;

    println!("\nartefacts written to {}", outdir.display());
    for (name, backend, bytes, _) in &manifest {
        println!("  {name:32} {backend:8} {bytes} bytes");
    }
    Ok(())
}
