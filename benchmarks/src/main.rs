//! Reproducible CPU benchmarks.
//!
//! # Reporting rules
//!
//! * Every number printed here is measured in this process. Nothing is
//!   estimated, extrapolated, or carried over from another machine.
//! * Each benchmark reports the workload size, the number of repetitions, the
//!   **median** wall-clock time per operation, and the fastest observed time.
//!   The median is used because it is robust to occasional scheduler noise.
//! * Throughput is derived from the median, and the derivation is stated.
//! * **No GPU or CUDA numbers are reported in M1**, because no GPU compute
//!   backend is implemented yet. Any claim of acceleration would be unsupported.
//!
//! # Usage
//!
//! ```text
//! cargo run --release -p atoms-benchmarks -- [repetitions]
//! ```
//!
//! Default repetitions: 7. Increase for a tighter interval.

use atoms_core::spatial_grid::SpatialGrid3D;
use atoms_numerics::sampling::{InverseCdf1D, SeededRng};
use atoms_orbitals::{DensityGrid, HydrogenWavefunction};
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;
use std::time::{Duration, Instant};

/// Run `f` `reps` times and return (median, min, total_samples).
fn measure<F: FnMut() -> u64>(reps: usize, mut f: F) -> (Duration, Duration, u64) {
    let mut times = Vec::with_capacity(reps);
    let mut work = 0u64;
    for _ in 0..reps {
        let t0 = Instant::now();
        work = f();
        times.push(t0.elapsed());
    }
    times.sort_unstable();
    let median = times[times.len() / 2];
    let min = times[0];
    (median, min, work)
}

fn fmt_dur(d: Duration) -> String {
    let ns = d.as_nanos() as f64;
    if ns < 1_000.0 {
        format!("{ns:.0} ns")
    } else if ns < 1_000_000.0 {
        format!("{:.2} us", ns / 1_000.0)
    } else if ns < 1_000_000_000.0 {
        format!("{:.2} ms", ns / 1_000_000.0)
    } else {
        format!("{:.3} s", ns / 1_000_000_000.0)
    }
}

fn cpu_description() -> String {
    if let Ok(out) = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
    {
        if out.status.success() {
            return String::from_utf8_lossy(&out.stdout).trim().to_string();
        }
    }
    if let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("model name") {
                return rest.trim_start_matches([':', ' ']).to_string();
            }
        }
    }
    "unknown CPU".to_string()
}

struct Row {
    name: String,
    workload: String,
    median: Duration,
    min: Duration,
    throughput: String,
}

fn print_table(rows: &[Row]) {
    let w0 = rows
        .iter()
        .map(|r| r.name.len())
        .max()
        .unwrap_or(10)
        .max(10);
    let w1 = rows
        .iter()
        .map(|r| r.workload.len())
        .max()
        .unwrap_or(10)
        .max(10);
    let w2 = 14;
    let w3 = 12;
    println!(
        "{:<w0$}  {:<w1$}  {:>w2$}  {:>w3$}  derived throughput",
        "benchmark", "workload", "median", "fastest"
    );
    println!("{}", "-".repeat(w0 + w1 + w2 + w3 + 24));
    for r in rows {
        println!(
            "{:<w0$}  {:<w1$}  {:>w2$}  {:>w3$}  {}",
            r.name,
            r.workload,
            fmt_dur(r.median),
            fmt_dur(r.min),
            r.throughput
        );
    }
}

fn main() {
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);

    println!("atoms-rs CPU benchmark");
    println!("======================");
    println!("cpu        : {}", cpu_description());
    println!(
        "platform   : {} / {} / {} rayon threads",
        std::env::consts::OS,
        std::env::consts::ARCH,
        rayon::current_num_threads()
    );
    println!("repetitions: {reps}");
    println!("backend    : cpu only (no GPU compute backend exists in M1)");
    println!("build      : release profile (lto, codegen-units = 1)");
    println!();
    println!("Timing method: wall clock around the whole kernel invocation, median of");
    println!("{reps} runs. Throughput = workload size / median time.");
    println!();

    let mut rows = Vec::new();

    // ---------------------------------------------------------------- radial
    {
        let states: Vec<(u32, u32)> = (1..=6).flat_map(|n| (0..n).map(move |l| (n, l))).collect();
        let radii: Vec<f64> = (0..256).map(|i| 0.05 + 0.05 * i as f64).collect();
        let (median, min, work) = measure(reps, || {
            let mut acc = 0.0;
            for &(n, l) in &states {
                for &r in &radii {
                    acc += atoms_orbitals::radial_wavefunction(n, l, r).unwrap();
                }
            }
            std::hint::black_box(acc);
            (states.len() * radii.len()) as u64
        });
        let per = median.as_secs_f64() / work as f64;
        rows.push(Row {
            name: "radial R_nl(r)".into(),
            workload: format!("{work} evals"),
            median,
            min,
            throughput: format!("{:.1} M evals/s", 1e-6 / per),
        });
    }

    // ------------------------------------------------- spherical harmonics
    {
        let states: Vec<(u32, i32)> = (0..=6u32)
            .flat_map(|l| {
                let lm = l as i32;
                (-lm..=lm).map(move |m| (l, m))
            })
            .collect();
        let angles: Vec<(f64, f64)> = (0..64)
            .map(|i| (0.05 + 0.045 * i as f64, 0.1 * i as f64))
            .collect();
        let (median, min, work) = measure(reps, || {
            let mut acc = 0.0;
            for &(l, m) in &states {
                for &(t, p) in &angles {
                    acc += atoms_orbitals::spherical_harmonics::evaluate(l, m, t, p).norm();
                }
            }
            std::hint::black_box(acc);
            (states.len() * angles.len()) as u64
        });
        let per = median.as_secs_f64() / work as f64;
        rows.push(Row {
            name: "spherical harmonic Y_l^m".into(),
            workload: format!("{work} evals"),
            median,
            min,
            throughput: format!("{:.1} M evals/s", 1e-6 / per),
        });
    }

    // -------------------------------------------------------- grid density
    for &n_grid in &[65usize, 129, 193] {
        let grid = SpatialGrid3D::new(
            Point3::new(-12.0, -12.0, -12.0),
            Vector3::new(
                24.0 / (n_grid as f64 - 1.0),
                24.0 / (n_grid as f64 - 1.0),
                24.0 / (n_grid as f64 - 1.0),
            ),
            [n_grid, n_grid, n_grid],
        )
        .unwrap();
        let wf = HydrogenWavefunction::new(3, 2, 0).unwrap();

        // parallel (rayon) path — the production path
        let (median_par, min_par, work) = measure(reps, || {
            let d = DensityGrid::probability_density(&wf, &grid);
            std::hint::black_box(d.max());
            d.len() as u64
        });
        let per_par = median_par.as_secs_f64() / work as f64;
        rows.push(Row {
            name: "density 3D grid (rayon)".into(),
            workload: format!("{} pts", grid.len()),
            median: median_par,
            min: min_par,
            throughput: format!("{:.1} M pts/s", 1e-6 / per_par),
        });

        // sequential path — reference for the parallel speedup claim
        let (median_seq, min_seq, _) = measure(reps, || {
            let mut acc = 0.0;
            for lin in 0..grid.len() {
                let p = grid.point_at(lin);
                acc += wf.probability_density(p.x, p.y, p.z);
            }
            std::hint::black_box(acc);
            grid.len() as u64
        });
        let per_seq = median_seq.as_secs_f64() / work as f64;
        rows.push(Row {
            name: "density 3D grid (1 thread)".into(),
            workload: format!("{} pts", grid.len()),
            median: median_seq,
            min: min_seq,
            throughput: format!("{:.1} M pts/s", 1e-6 / per_seq),
        });
        let speedup = median_seq.as_secs_f64() / median_par.as_secs_f64();
        println!(
            "measured parallel speedup at {}^3 = {} points: {:.2}x on {} threads",
            n_grid,
            grid.len(),
            speedup,
            rayon::current_num_threads()
        );
    }

    // ------------------------------------------------------------- sampling
    {
        let wf = HydrogenWavefunction::new(2, 1, 0).unwrap();
        let sampler =
            InverseCdf1D::from_pdf(|r| wf.radial_probability_density(r), 0.0, 60.0, 20_000)
                .unwrap();
        let n_samples = 1_000_000u64;
        let (median, min, _) = measure(reps, || {
            let mut rng = SeededRng::from_seed(20261015);
            let mut acc = 0.0;
            for _ in 0..n_samples {
                acc += sampler.sample(&mut rng);
            }
            std::hint::black_box(acc);
            n_samples
        });
        let per = median.as_secs_f64() / n_samples as f64;
        rows.push(Row {
            name: "radial sampling (1 stream)".into(),
            workload: format!("{n_samples} samples"),
            median,
            min,
            throughput: format!("{:.1} M samples/s", 1e-6 / per),
        });

        // Parallel independent streams: seeded per index so the result is
        // reproducible regardless of thread scheduling.
        let (median_par, min_par, _) = measure(reps, || {
            (0..8u64)
                .into_par_iter()
                .map(|chunk| {
                    let mut rng = SeededRng::from_seed(20261015 + chunk);
                    let mut acc = 0.0;
                    for _ in 0..(n_samples / 8) {
                        acc += sampler.sample(&mut rng);
                    }
                    acc
                })
                .sum::<f64>();
            n_samples
        });
        let per_par = median_par.as_secs_f64() / n_samples as f64;
        rows.push(Row {
            name: "radial sampling (8 streams)".into(),
            workload: format!("{n_samples} samples"),
            median: median_par,
            min: min_par,
            throughput: format!("{:.1} M samples/s", 1e-6 / per_par),
        });
    }

    print_table(&rows);

    // ------------------------------------------------------------- memory
    println!();
    println!("Memory footprint of one density buffer (8 bytes per f64 point):");
    for &n_grid in &[65usize, 129, 193, 257] {
        let bytes = (n_grid * n_grid * n_grid) * 8;
        println!(
            "  {:>3}^3 = {:>11} points -> {:>8.1} MiB",
            n_grid,
            n_grid * n_grid * n_grid,
            bytes as f64 / (1024.0 * 1024.0)
        );
    }
    println!();
    println!("No GPU or CUDA figures are reported: no GPU compute backend is");
    println!("implemented in M1. See crate `atoms-gpu` (target milestone M5).");
}
