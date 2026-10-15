# atoms-rs

A scientifically validated quantum physics and molecular research framework in
Rust, with a Python scripting interface and a planned GPU-accelerated viewer.

**Current status: M2-A complete** (Python bindings). The Rust core (M1) and the
Python API are implemented and tested. The GPU renderer and the desktop viewer
are **not implemented** — see [Status](#status).

```text
cargo test --workspace                   ->  183 passed; 0 failed
cargo clippy --workspace --all-targets   ->  0 warnings
.venv/bin/python -m pytest python/tests  ->   53 passed; 0 failed
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

The `atoms` Python package exposes all of it, with NumPy array transfer.

## Quick start (Python)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install maturin pytest numpy scipy
cd crates/atoms-python && ../../.venv/bin/maturin develop --release
```

```python
import atoms

hydrogen = atoms.Atom.hydrogen()
orbital = hydrogen.orbital(n=2, l=1, m=0)          # 2p_z

grid = atoms.Grid3D(extent=(-14.0, 14.0), resolution=(141, 141, 141), unit="bohr")
density = orbital.probability_density(grid)        # float64 (141, 141, 141)

print(orbital.energy("ev"))                        # -3.401423281 eV
print(density.sum() * grid.volume_element_bohr3)   # 0.999599

positions = atoms.sample_positions(orbital, 20_000, seed=20261015)
```

See `docs/python-api.md` for the full contract (units, axis order, dtypes,
exception behaviour).

## Quick start (Rust, no GPU and no Python required)

```bash
cargo test --workspace
cargo run --release -p atoms-benchmarks
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

## Numerical validation

Every number below is measured in this repository, against implementations that
share no code with it.

| quantity | reference | samples | measured deviation |
|---|---|---|---|
| `Y_l^m` | `scipy.special.sph_harm_y` | 252 | 1.2e-15 abs (Python API) |
| `R_{nl}` | `scipy.special.assoc_laguerre` | 288 | 4.7e-15 rel |
| `\|psi\|^2` | SciPy-derived | 108 | 6.8e-15 rel (Python API) |
| `L_k^a` | **exact rational arithmetic** | 84 | 1.0e-14 rel |

Plus analytic checks: energies against `-Z^2/2n^2` and the virial theorem,
Lyman-alpha and Balmer-alpha wavelengths (with and without the reduced-mass
correction), radial node count `n-l-1`, angular node count `l`, `2p` nodal sets
for `m=0` vs `|m|=1`, radial Schrodinger / Legendre / Laguerre differential
equation residuals, harmonic orthonormality and the addition theorem, marginal
normalisation, and sampler moments against quadrature.

## Status

| milestone | scope | status |
|---|---|---|
| M0 | repository and architecture audit | complete |
| M1 | validated atomic core | complete — `docs/validation/m1-report.md` |
| M2-A | Python bindings and NumPy transfer | **complete** |
| M2-B | portable `wgpu` rendering foundation | not started |
| M2-C | interactive orbital explorer | not started |
| M2-D | CI and platform validation | not started |
| M3 | interactions and molecular geometry | not started |
| M4 | quantum chemistry (Hartree-Fock, H2) | not started |
| M5 | GPU compute and advanced visualisation | not started |

The following crates are **documented stubs** with no functionality. Each states
its target milestone and its contract in its own source:

| crate / app | target | contains |
|---|---|---|
| `atoms-interactions` | M3 | placeholder |
| `atoms-molecule` | M3 | placeholder |
| `atoms-chemistry` | M4 | placeholder |
| `atoms-gpu` | M5 | placeholder |
| `atoms-render` | M2-B | placeholder |
| `apps/atoms-viewer` | M2-C | prints status and exits |

`atoms.viewer.show(...)` exists in the Python API but raises
`NotImplementedError`; the viewer is M2-C.

**No molecular property is computed anywhere in this workspace.** No total
energy, binding energy, dipole moment, or density for any molecule. No
Hartree-Fock, no DFT, no basis sets. No GPU or CUDA kernel exists, and no GPU
performance claim is made.

## Layout

```text
atoms-rs/
├── crates/
│   ├── atoms-core/          # units, coordinates, grids, species, energy levels
│   ├── atoms-numerics/      # special functions, quadrature, sampling
│   ├── atoms-orbitals/      # wavefunctions, harmonics, densities, sampling
│   ├── atoms-interactions/  # M3 stub
│   ├── atoms-io/            # reproducibility metadata (JSON)
│   ├── atoms-molecule/      # M3 stub
│   ├── atoms-chemistry/     # M4 stub
│   ├── atoms-gpu/           # M5 stub
│   ├── atoms-render/        # M2-B stub
│   └── atoms-python/        # PyO3 extension module `atoms._atoms`
│       ├── src/             #   bindings + NumPy ordering (Rust-tested)
│       ├── python/atoms/    #   Python front end (docstrings, namespaces)
│       └── pyproject.toml   #   maturin config
├── apps/atoms-viewer/       # M2-C stub
├── benchmarks/              # measurement harness
├── tests/validation/        # cross-crate scientific validation
├── python/
│   ├── examples/            # reproducible examples
│   └── tests/               # pytest suites + reference generator
└── docs/
    ├── architecture/        # workspace design
    ├── physics/             # normative conventions
    ├── numerical-methods/   # methods and measured accuracy
    ├── python-api.md        # Python contract
    └── validation/          # reference data, benchmarks, milestone reports
```

## Conventions

Read `docs/physics/conventions.md` before using any API. In short: **Hartree
atomic units internally**, physics convention for angles (`theta` polar from
`+z`), `|psi|^2` and `r^2|R|^2` are different quantities with different units,
and negative `m` is fully supported with an explicit complex-phase convention.

Python callers: `out[i, j, k]` is the density at `(x_i, y_j, z_k)`, array shape
`(nx, ny, nz)`, `float64`, C order.

## Reproducing the reference data

```bash
.venv/bin/python python/tests/generate_reference_values.py
```

This regenerates `docs/validation/reference-values.csv` (732 values). The file is
committed so the Rust test suite needs no Python environment, and the values are
**generated, never transcribed by hand**.

## Attribution

Inspired by [kavan010/Atoms](https://github.com/kavan010/Atoms). That repository
was read only as a reference and is **not** a dependency; no code was copied, and
its physics approximations are not inherited. See
`docs/architecture/atoms-rs-design.md`.

## License

MIT.
