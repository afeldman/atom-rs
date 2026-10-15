# atoms-rs

A scientifically validated quantum physics and molecular research framework in
Rust, with a planned Python scripting interface and GPU-accelerated
visualisation.

**Current status: M1 complete.** The validated one-electron atomic core is
implemented and tested. Molecular electronic structure, the Python bindings, and
the viewer are **not implemented** — see [Status](#status) below.

```text
cargo test --workspace     ->  168 passed; 0 failed
cargo clippy --workspace --all-targets  ->  0 warnings
```

## What is implemented

`atoms-core`, `atoms-numerics`, `atoms-orbitals` implement the exact analytic
solution of the hydrogen-like Schrodinger equation:

* energies `E_n = -Z^2/(2n^2)` hartree, with the virial relation and reduced-mass
  correction;
* radial wavefunctions `R_{nl}(r)` with generalised Laguerre polynomials;
* complex spherical harmonics `Y_l^m(theta, phi)` with the Condon-Shortley phase,
  including correct behaviour for negative `m`;
* the full wavefunction, probability density `|psi|^2`, and radial probability
  density `r^2|R|^2` as **separate, separately documented quantities**;
* special functions, composite quadrature, and reproducible seeded sampling;
* volumetric evaluation on Cartesian grids with 2D cross-sections;
* analytic Kramers-Pasternack expectation values as validation references.

Accuracy is verified against independent implementations:

| quantity | reference | measured max deviation |
|---|---|---|
| `Y_l^m` | `scipy.special.sph_harm`, 252 samples | `4.4e-16` |
| `R_{nl}` | `scipy.special.assoc_laguerre`, 288 samples | `4.7e-15` |
| `|psi|^2` | SciPy-derived, 108 samples | `1.4e-15` |
| `L_k^a` | **exact rational arithmetic**, 84 evaluations | `1.0e-14` |

## Status

| milestone | scope | status |
|---|---|---|
| M0 | repository and architecture audit | complete |
| M1 | validated atomic core | **complete** — see `docs/validation/m1-report.md` |
| M2 | Python API and first 3D viewer | not started |
| M3 | interactions and molecular geometry | not started |
| M4 | quantum chemistry (Hartree-Fock, H2) | not started |
| M5 | GPU compute and advanced visualisation | not started |

The following crates are present as **documented stubs** and contain no
functionality. Each states its target milestone and its contract in its own
source:

| crate | target | contains |
|---|---|---|
| `atoms-interactions` | M3 | placeholder |
| `atoms-molecule` | M3 | placeholder |
| `atoms-chemistry` | M4 | placeholder |
| `atoms-gpu` | M5 | placeholder |
| `atoms-render` | M2 | placeholder |
| `atoms-python` | M2 | placeholder |
| `apps/atoms-viewer` | M2 | prints status and exits |

In particular: **no molecular property is computed anywhere in this workspace.**
No total energy, binding energy, dipole moment, orbital, or density for any
molecule. No Hartree-Fock, no DFT, no basis sets.

## Quick start

```bash
cargo test --workspace           # 168 tests, no GPU or Python needed
cargo run --release -p atoms-benchmarks   # measured CPU benchmarks
```

```rust
use atoms_core::spatial_grid::SpatialGrid3D;
use atoms_orbitals::{DensityGrid, HydrogenWavefunction};
use nalgebra::{Point3, Vector3};

let wf = HydrogenWavefunction::new(2, 1, 0).unwrap();   // 2p_z
let grid = SpatialGrid3D::new(
    Point3::new(-12.0, -12.0, -12.0),
    Vector3::new(0.125, 0.125, 0.125),
    [193, 193, 193],
).unwrap();
let density = DensityGrid::probability_density(&wf, &grid);
// measured on this grid: 0.997943, the remainder being the truncated outer tail
// plus the O(dx^2) Riemann error -- the library never renormalises
assert!(density.total_probability() > 0.99);
```

This exact snippet is executed by the test suite
(`integration.rs::readme_quickstart_snippet_runs`), so the documentation cannot
silently rot.

## Layout

```text
atoms-rs/
├── crates/
│   ├── atoms-core/          # units, coordinates, grids, species, energy levels
│   ├── atoms-numerics/      # special functions, quadrature, sampling
│   ├── atoms-orbitals/      # wavefunctions, harmonics, densities
│   ├── atoms-interactions/  # M3 stub
│   ├── atoms-io/            # reproducibility metadata (JSON)
│   ├── atoms-molecule/      # M3 stub
│   ├── atoms-chemistry/     # M4 stub
│   ├── atoms-gpu/           # M5 stub
│   ├── atoms-render/        # M2 stub
│   └── atoms-python/        # M2 stub
├── apps/atoms-viewer/       # M2 stub
├── benchmarks/              # measurement harness
├── tests/validation/        # cross-crate scientific validation
├── python/tests/            # independent reference generator
└── docs/
    ├── architecture/        # workspace design
    ├── physics/             # normative conventions
    ├── numerical-methods/   # methods and measured accuracy
    └── validation/          # reference data, benchmarks, milestone reports
```

## Conventions

Read `docs/physics/conventions.md` before using any API. In short: **Hartree
atomic units internally**, physics convention for angles (`theta` polar from
`+z`), `|psi|^2` and `r^2|R|^2` are different quantities with different units,
and negative `m` is fully supported with an explicit complex-phase convention.

## Reproducing the reference data

```bash
python -m venv .venv && .venv/bin/pip install numpy scipy
.venv/bin/python python/tests/generate_reference_values.py
```

This regenerates `docs/validation/reference-values.csv`. The file is committed so
that the Rust test suite needs no Python environment, and the values are
generated rather than transcribed.

## Attribution

Inspired by [kavan010/Atoms](https://github.com/kavan010/Atoms). That repository
was read only as a reference and is **not** a dependency; no code was copied, and
its physics approximations are not inherited. See
`docs/architecture/atoms-rs-design.md` for what was taken as inspiration, what
was replaced, and why.

## License

MIT.
