//! End-to-end pipelines and architectural invariants.
//!
//! The architectural invariants are real tests rather than documentation: they
//! fail if someone later wires a GPU, GUI, or Python dependency into the physics
//! core, which is the single most important boundary in this workspace.

use atoms_core::spatial_grid::SpatialGrid3D;
use atoms_io::CalculationMetadata;
use atoms_orbitals::{DensityGrid, HydrogenWavefunction};
use nalgebra::{Point3, Vector3};
use std::f64::consts::PI;

fn small_volumetric_grid() -> SpatialGrid3D {
    SpatialGrid3D::new(
        Point3::new(-10.0, -10.0, -10.0),
        Vector3::new(0.25, 0.25, 0.25),
        [81, 81, 81],
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Architectural invariants
// ---------------------------------------------------------------------------

/// The physics core must not depend on graphics, GPU, or Python crates.
#[test]
fn core_crates_have_no_graphics_gpu_or_python_dependencies() {
    // The manifests are compiled into the test binary, so a regression that adds
    // such a dependency fails this test rather than only being noticeable in a
    // code review.
    let core = include_str!("../../../crates/atoms-core/Cargo.toml");
    let numerics = include_str!("../../../crates/atoms-numerics/Cargo.toml");
    let orbitals = include_str!("../../../crates/atoms-orbitals/Cargo.toml");

    for (name, manifest) in [
        ("atoms-core", core),
        ("atoms-numerics", numerics),
        ("atoms-orbitals", orbitals),
    ] {
        for forbidden in [
            "wgpu",
            "pyo3",
            "eframe",
            "egui",
            "atoms-render",
            "atoms-gpu",
            "atoms-python",
            "cudarc",
            "vulkan",
            "ash",
        ] {
            assert!(
                !manifest.contains(forbidden),
                "{name} must not depend on `{forbidden}`"
            );
        }
    }
}

/// The workspace must contain every crate named in the architecture document.
#[test]
fn workspace_declares_the_planned_members() {
    let root = include_str!("../../../Cargo.toml");
    for member in [
        "crates/atoms-core",
        "crates/atoms-numerics",
        "crates/atoms-orbitals",
        "crates/atoms-interactions",
        "crates/atoms-io",
        "crates/atoms-molecule",
        "crates/atoms-chemistry",
        "crates/atoms-gpu",
        "crates/atoms-render",
        "crates/atoms-python",
        "apps/atoms-viewer",
        "tests/validation",
        "benchmarks",
    ] {
        assert!(
            root.contains(member),
            "workspace is missing member `{member}`"
        );
    }
}

/// Crates that are declared but not implemented must say so in their own source,
/// so that no reader can mistake a placeholder for a working model — and crates
/// that *are* implemented must not claim to be placeholders.
///
/// `atoms-render` moved from the first list to the second in M2-B; this test is
/// what forced that bookkeeping to be explicit rather than left stale.
#[test]
fn unimplemented_crates_declare_their_status() {
    let stubs: [(&str, &str, &str); 4] = [
        (
            "atoms-interactions",
            include_str!("../../../crates/atoms-interactions/src/lib.rs"),
            "M3",
        ),
        (
            "atoms-molecule",
            include_str!("../../../crates/atoms-molecule/src/lib.rs"),
            "M3",
        ),
        (
            "atoms-chemistry",
            include_str!("../../../crates/atoms-chemistry/src/lib.rs"),
            "M4",
        ),
        (
            "atoms-gpu",
            include_str!("../../../crates/atoms-gpu/src/lib.rs"),
            "M5",
        ),
    ];
    for (name, source, milestone) in stubs {
        assert!(
            source.contains("not yet implemented"),
            "{name} must be marked as not yet implemented"
        );
        assert!(
            source.contains(milestone),
            "{name} must name its target milestone ({milestone})"
        );
    }

    let implemented: [(&str, &str); 5] = [
        (
            "atoms-core",
            include_str!("../../../crates/atoms-core/src/lib.rs"),
        ),
        (
            "atoms-numerics",
            include_str!("../../../crates/atoms-numerics/src/lib.rs"),
        ),
        (
            "atoms-orbitals",
            include_str!("../../../crates/atoms-orbitals/src/lib.rs"),
        ),
        (
            "atoms-render",
            include_str!("../../../crates/atoms-render/src/lib.rs"),
        ),
        (
            "atoms-python",
            include_str!("../../../crates/atoms-python/src/lib.rs"),
        ),
    ];
    for (name, source) in implemented {
        assert!(
            !source.contains("not yet implemented"),
            "{name} is implemented and must not declare itself a stub"
        );
    }
}

// ---------------------------------------------------------------------------
// End-to-end pipelines
// ---------------------------------------------------------------------------

/// Orbital -> grid -> density -> cross-section -> metadata JSON.
#[test]
fn orbital_to_density_to_metadata_pipeline() {
    let grid = small_volumetric_grid();
    let wf = HydrogenWavefunction::new(2, 1, 0).unwrap();
    let density = DensityGrid::probability_density(&wf, &grid);

    assert_eq!(density.len(), grid.len());
    assert!(density.max() > 0.0);

    // 2p_z must have zero density on the mid-plane.
    let [nx, ny, nz] = grid.counts();
    let mid = density.cross_section_xy(nz / 2);
    assert!(mid.iter().all(|&v| v < 1e-25));

    // The slice must be a view of the same data, not a copy with a different
    // orientation.
    for j in 0..ny.min(5) {
        for i in 0..nx.min(5) {
            assert_eq!(mid[j * nx + i], density.value_at(i, j, nz / 2));
        }
    }

    let metadata = CalculationMetadata::new(
        "hydrogenic-1e (unscreened, infinite nuclear mass)",
        "bohr, hartree",
        env!("CARGO_PKG_VERSION"),
    )
    .with_parameter("grid_counts", format!("{nx}x{ny}x{nz}"))
    .with_parameter("grid_spacing_bohr", format!("{}", grid.spacing().x))
    .with_backend("cpu")
    .with_notes("no screening, no electron correlation, no relativistic corrections");

    let json = metadata.to_json().expect("serialise metadata");
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid json");
    assert_eq!(parsed["units"], "bohr, hartree");
    assert_eq!(parsed["backend"], "cpu");
    assert!(parsed["basis_set"].is_null(), "no basis set is used");
    assert!(
        parsed["convergence"].is_null(),
        "the model is not iterative"
    );
    assert!(parsed["random_seed"].is_null(), "the grid is deterministic");
    let notes = parsed["notes"].as_str().unwrap();
    assert!(notes.contains("no screening"));
}

/// Grid evaluation is parallelised; it must nevertheless be deterministic.
#[test]
fn parallel_density_evaluation_is_bitwise_deterministic() {
    let grid = small_volumetric_grid();
    let wf = HydrogenWavefunction::new(3, 2, -1).unwrap();
    let a = DensityGrid::probability_density(&wf, &grid);
    let b = DensityGrid::probability_density(&wf, &grid);
    assert_eq!(
        a.values(),
        b.values(),
        "parallel reduction is not deterministic"
    );
}

/// A reproducible experiment: same seed and same parameters give the same
/// trajectory of samples, which is the prerequisite for reproducible science.
#[test]
fn seeded_experiment_is_reproducible_and_metadata_records_the_seed() {
    use atoms_numerics::sampling::{InverseCdf1D, SeededRng};
    use atoms_orbitals::radial_wavefunction;

    let pdf = |r: f64| {
        let v = radial_wavefunction(1, 0, r).unwrap();
        r * r * v * v
    };
    let sampler = InverseCdf1D::from_pdf(pdf, 0.0, 40.0, 20_000).unwrap();
    let run = |seed: u64| -> Vec<f64> {
        let mut rng = SeededRng::from_seed(seed);
        (0..1000).map(|_| sampler.sample(&mut rng)).collect()
    };
    let first = run(20261015);
    let second = run(20261015);
    assert_eq!(first, second);
    assert_ne!(first, run(20261016));

    let metadata = CalculationMetadata::new("hydrogenic-1e sampling", "bohr", "0.1.0")
        .with_seed(20261015)
        .with_parameter("samples", "1000");
    let json = metadata.to_json().unwrap();
    assert!(json.contains("20261015"), "the seed must be recorded");
}

/// The angular and radial marginals must jointly reproduce unit normalisation,
/// which is the end-to-end statement that the two independent code paths
/// (quadrature over `r` and quadrature over `theta`) are mutually consistent.
#[test]
fn marginal_normalisations_are_mutually_consistent() {
    use atoms_numerics::quadrature::simpson;
    for (n, l, m) in [
        (1u32, 0u32, 0i32),
        (2, 1, 1),
        (3, 2, -2),
        (4, 0, 0),
        (5, 3, 3),
    ] {
        let wf = HydrogenWavefunction::new(n, l, m).unwrap();
        let radial = simpson(
            |r| wf.radial_probability_density(r),
            0.0,
            12.0 * (n * n) as f64 + 40.0,
            200_000,
        );
        let angular = simpson(
            |t| wf.angular_probability_density(t) * t.sin(),
            0.0,
            PI,
            200_000,
        );
        let total = radial * angular * 2.0 * PI;
        assert!(
            (total - 1.0).abs() < 1e-8,
            "n={n} l={l} m={m}: radial={radial} angular={angular} total={total}"
        );
    }
}

/// Grid metadata must round-trip through serde, so that a saved grid can be
/// reloaded and re-evaluated identically.
#[test]
fn grid_round_trips_through_serde() {
    let grid =
        SpatialGrid3D::centred(Point3::new(0.5, -1.0, 2.0), [8.0, 4.0, 2.0], [17, 9, 5]).unwrap();
    let json = serde_json::to_string(&grid).unwrap();
    let back: SpatialGrid3D = serde_json::from_str(&json).unwrap();
    assert_eq!(grid, back);
    assert_eq!(grid.counts(), back.counts());
}

/// The "Quick start" snippet in `README.md` must actually run. Documentation
/// that is not executed rots; this test is the execution.
#[test]
fn readme_quickstart_snippet_runs() {
    let wf = HydrogenWavefunction::new(2, 1, 0).unwrap(); // 2p_z
    let grid = SpatialGrid3D::new(
        Point3::new(-12.0, -12.0, -12.0),
        Vector3::new(0.125, 0.125, 0.125),
        [193, 193, 193],
    )
    .unwrap();
    let density = DensityGrid::probability_density(&wf, &grid);
    let total = density.total_probability();
    println!(
        "README quickstart: 2p_z on a 193^3 grid over [-12,12] bohr -> \
         int|psi|^2 dV = {total:.6}"
    );
    assert!(density.total_probability() > 0.99);

    // The README is part of the repository and is checked, not assumed, to
    // contain this snippet.
    let readme = include_str!("../../../README.md");
    assert!(readme.contains("HydrogenWavefunction::new(2, 1, 0)"));
    assert!(readme.contains("density.total_probability() > 0.99"));
}
