# atoms-rs Architecture Design Document

> **Repository:** `atom-rs`  
> **Version:** M0 — Initial Architecture (2026-10-15)  
> **Status:** Design Complete — Pending User Approval  

---

## Executive Summary

`atoms-rs` is a scientifically validated quantum physics and molecular research framework in Rust, inspired by `kavan010/Atoms`. The goal is **not** to port the existing C++ OpenGL application but to build a reusable, extensible, scientifically validated library for:

- Hydrogen atom quantum mechanics (exact analytical solutions)
- Atomic models with proper nuclear/electronic representation
- Interatomic interactions via documented approximate potentials
- Molecular geometry and property computation
- GPU-accelerated visualization (Vulkan/Metal via `wgpu`)
- Python scripting interface (PyO3)

**Core Principle:** Scientific correctness > visual demonstration. Every calculation must be validated against analytical solutions or established references.

---

## 1. Reference Repository Analysis Summary

### Functionality Ported (Adapted, Not Reproduced):

| Original | atoms-rs Approach |
|----------|-------------------|
| `atom.cpp` — 2D Bohr model simulation | **Not ported** — replaced by quantum orbital visualization |
| `atom_raytracer.cpp` — Ray-traced orbital sampling | **Reimplemented** — GPU-accelerated via wgpu + WGSL shaders |
| `atom_realtime.cpp` — Realtime CDF-sampled points | **Reimplemented** — CPU/GPU kernels with validation |
| `wave_atom_2d.cpp` — Wave visualization | **Not ported** — replaced by physics-informed visualization |
| `schrodinger.py` — Orbital sampling | **Reimplemented** — Rust core + Python bindings via PyO3 |

### Mathematical Models Preserved:

1. **Hydrogen eigenstates:** ψ(r,θ,φ) = R_{n,l}(r) · Y_l^m(θ,φ)
2. **Radial wavefunction:** R(n,l,r) with associated Laguerre polynomials
3. **Spherical harmonics:** Complex Y_l^m via scipy.special (or equivalent)
4. **Energy eigenvalues:** E_n = -13.6 eV / n² (atomic units: -0.5/n² Ha)
5. **Probability density:** |ψ|² and radial probability r²|R|²

### Models Discarded/Replaced:

- Simple classical particle repulsion (Coulomb-only, not proper QM)
- Hard-coded quantum numbers without validation
- No normalization checks
- Approximate multi-electron model (not Hartree-Fock or DFT)

---

## 2. Architecture Overview

### Workspace Structure:

```text
atom-rs/
├── Cargo.toml                 # Workspace manifest
├── crates/
│   ├── atoms-core/           # Pure physics: hydrogen atom, units, coordinates
│   ├── atoms-numerics/       # Numerical methods: integration, sampling
│   ├── atoms-orbitals/       # Wavefunctions, spherical harmonics, densities
│   ├── atoms-interactions/    # Potentials: Coulomb, Lennard-Jones, etc.
│   ├── atoms-molecule/       # Molecular geometry, basic properties
│   ├── atoms-chemistry/      # Electronic structure solvers (M3+)
│   ├── atoms-gpu/            # GPU kernels (optional CUDA backend)
│   ├── atoms-render/         # wgpu shaders, geometry rendering
│   ├── atoms-io/             # XYZ, JSON, NumPy-compatible exports
│   └── atoms-python/         # PyO3 bindings, NumPy interoperability
├── apps/
│   └── atoms-viewer/         # egui + wgpu desktop application
├── python/
│   ├── examples/             # Python examples
│   └── tests/                # Python binding tests
├── benchmarks/               # CPU/GPU performance benchmarks
├── docs/
│   ├── architecture/
│   ├── physics/
│   └── validation/
└── tests/
    ├── physics/              # Unit tests for quantum mechanics
    ├── numerical/            # Numerical integration tests
    └── integration/          # End-to-end system tests
```

### Design Principles:

1. **Separation of concerns:** Physics core has zero GPU/rendering dependencies.
2. **Validation first:** Every calculation includes unit tests against analytical solutions.
3. **GPU optional:** Base library works without graphics hardware.
4. **Explicit units:** All quantities track units (atomic, SI, etc.) with conversions.
5. **Extensible solvers:** Electronic structure methods defined via trait-based interface.
6. **Metadata-rich output:** Every export includes model, version, convergence info.

---

## 3. Public API Design (Rust & Python)

### Rust Core API:

```rust
use atoms_core::{Atom, Units};
use atoms_orbitals::{HydrogenOrbital, ProbabilityDensity};
use atoms_io::{ExportXYZ, ExportJSON};

// Create hydrogen atom
let hydrogen = Atom::hydrogen();

// Create orbital (n=2, l=1, m=0)
let orbital = HydrogenOrbital::new(2, 1, 0)?;

// Create spatial grid
let grid = SpatialGrid3D::new(
    extent: (-10.0, 10.0, 10.0, 10.0, 10.0), // (x_min, x_max, ...)
    resolution: (128, 128, 128),
    units: Units::bohr,
)?;

// Compute probability density
let density = ProbabilityDensity::compute(&orbital, &grid)?;

// Export to XYZ or JSON
hydrogen.export_xyz("molecule.xyz")?;
```

### Python API (via PyO3):

```python
import atoms
import numpy as np

# Hydrogen atom
hydrogen = atoms.Atom.hydrogen()

# Orbital (n=2, l=1, m=0)
orbital = hydrogen.orbital(n=2, l=1, m=0)

# 3D grid
grid = atoms.Grid3D(
    extent=(-10.0, 10.0, 10.0, 10.0),
    resolution=(128, 128, 128),
    unit="bohr",
)

# Probability density
density = orbital.probability_density(grid)

# Visualize
atoms.viewer.show(orbital=orbital, quantity="probability_density")

# Export
atoms.io.save_grid(density, "density.npy")
```

---

## 4. Milestone Breakdown

### M0 — Repository and Architecture Audit (Current)

**Deliverables:**
- [x] Reference repository inspection
- [ ] Design document (this document)
- [ ] Workspace initialization
- [ ] CI/CD pipeline setup
- [ ] Scientific reference cases documented
- [ ] Initial public APIs defined

### M1 — Validated Atomic Core

**Deliverables:**
- [ ] Hydrogen atom implementation (`atoms-core`)
  - Energy eigenvalues (E_n = -0.5/n² Ha)
  - Radial wavefunctions R_{n,l}(r)
  - Spherical harmonics Y_l^m(θ,φ)
  - Complex wavefunctions ψ(r,θ,φ)
- [ ] Spatial grid and coordinate transformations
- [ ] Unit system (atomic units as default, SI conversion)
- [ ] Validation tests:
  - Energy eigenvalues vs analytical E_n = -0.5/n² Ha
  - Wavefunction normalization: ∫|ψ|² dV = 1
  - Probability density non-negativity
  - Nodal structure verification (e.g., s-orbitals nodeless, p-orbitals have nodal planes)

### M2 — Python API and First 3D Viewer

**Deliverables:**
- [ ] PyO3 bindings (`atoms-python`)
- [ ] wgpu-based renderer (`atoms-render` + `apps/atoms-viewer`)
- [ ] Orbital visualization (isosurfaces, color maps)
- [ ] Metal backend verification (macOS)
- [ ] Vulkan backend verification (Linux)

### M3 — Interactions and Molecular Infrastructure

**Deliverables:**
- [ ] Atomic species database (Z, mass, config)
- [ ] Lennard-Jones potential with force gradients
- [ ] Simple molecular geometry builder
- [ ] Energy-distance curve plotting
- [ ] Basic dipole moment calculation

### M4 — Quantum Chemistry

**Deliverables:**
- [ ] H₂ minimal model (two-center basis, analytical integrals)
- [ ] Hartree-Fock solver (M3+ milestone, optional)
- [ ] Molecular orbital and density evaluation
- [ ] Solver convergence diagnostics

### M5 — GPU Compute and Advanced Visualization

**Deliverables:**
- [ ] wgpu-based electron density volume rendering
- [ ] Optional CUDA backend for large-grid computations
- [ ] Simulation trajectory visualization
- [ ] Performance benchmarks (CPU vs. GPU)

---

## 5. Scientific Reference Cases

### Hydrogen Atom Validation Tests:

1. **Ground state energy:** E_1 = -0.5 Ha = -27.2114 eV
2. **n=2 energy:** E_2 = -0.5/4 = -0.125 Ha = -6.8029 eV
3. **1s wavefunction at r=0:** ψ(0) ≠ 0; at r=a₀, maximum probability
4. **2p orbital nodal plane:** m=0 has planar node at θ=π/2

### Numerical Validation:

- Probability density must be ≥ 0 everywhere
- Radial probability P(r) = r²|R|² must integrate to 1
- Spherical harmonics orthonormality: ∫Y*_{l'm'} Y_{lm} dΩ = δ_{ll'}δ_{mm'}

---

## 6. Next Steps (M0 Completion)

Before proceeding to M1 implementation:

1. **User approval** of this architectural design
2. **CI/CD setup** (GitHub Actions for formatting, linting, tests)
3. **Scientific validation baseline** documented in `docs/validation/`
4. **Workspace initialization** with proper dependency versions pinned

---

## 7. Known Limitations & Deferred Work

- Multi-electron atoms: deferred to electronic structure solvers (M4+)
- Density Functional Theory: deferred until Hartree-Fock validated
- Native Vulkan backend (ash): optional, evaluated after wgpu baseline exists
- CUDA compute: optional backend with CPU fallback
- Complete atomic databases: start with hydrogen/helium, extend as needed

---

## 8. Conclusion

This architecture prioritizes scientific correctness over visual demonstration. Every milestone includes validation tests and documentation. GPU acceleration is optional but well-supported. The Python interface enables interactive experimentation while the Rust core remains independent.

**Ready for user review and approval.**
