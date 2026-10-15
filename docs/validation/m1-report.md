# M1 — Validated atomic core: completion report

**Status: complete, pending your review.**
**Acceptance gates: all verified.** Evidence for each is listed below.

---

## 1. Changes made

### 1.1 New crates and files

| file | lines | content |
|---|---|---|
| `crates/atoms-numerics/src/special.rs` | 279 | `factorial`, `double_factorial`, `generalized_laguerre`, `associated_legendre_cs`, `associated_legendre_no_cs` |
| `crates/atoms-numerics/src/quadrature.rs` | 111 | composite Simpson, trapezoid, semi-infinite Simpson |
| `crates/atoms-numerics/src/sampling.rs` | 313 | `SeededRng` (ChaCha8), `InverseCdf1D`, rejection sampling |
| `crates/atoms-core/src/units.rs` | 248 | atomic-unit constants (CODATA 2018), `LengthUnit`, `EnergyUnit`, conversions |
| `crates/atoms-core/src/coordinates.rs` | 177 | Cartesian <-> spherical, physics convention |
| `crates/atoms-core/src/spatial_grid.rs` | 343 | `SpatialGrid3D`, validated construction, index layout, `map`/`integrate` |
| `crates/atoms-core/src/hydrogen_energy.rs` | 301 | `EnergyLevel`, reduced-mass ratio, transition wavelengths |
| `crates/atoms-core/src/atom.rs` | 192 | `AtomicSpecies` (Z = 1..10), `Atom` |
| `crates/atoms-orbitals/src/quantum_numbers.rs` | 281 | `QuantumNumbers`, validated, negative `m` |
| `crates/atoms-orbitals/src/radial.rs` | 404 | `R_{nl}(r)`, radial probability density, `Z` scaling |
| `crates/atoms-orbitals/src/spherical_harmonics.rs` | 490 | complex `Y_l^m`, Condon-Shortley phase, `m < 0` relation |
| `crates/atoms-orbitals/src/wavefunction.rs` | 400 | `HydrogenWavefunction`, spherical and Cartesian evaluation |
| `crates/atoms-orbitals/src/density.rs` | 444 | `DensityGrid`, cross-sections, `radial_expectation_value` |
| `crates/atoms-orbitals/src/expectations.rs` | 123 | analytic closed forms (Kramers-Pasternack) |
| `crates/atoms-io/src/lib.rs` | 140 | `CalculationMetadata` with JSON round trip |
| `tests/validation/{src/lib.rs,tests/*}` | 1304 | reference-value parsing, physics/numerical/integration validation |
| `benchmarks/src/main.rs` | 300 | measurement harness |
| `python/tests/generate_reference_values.py` | 233 | independent reference generator (SciPy + exact rational arithmetic) |

Total Rust source: **6 223 lines** across the workspace.

### 1.2 Documented stubs (no functionality, status declared in-source)

`atoms-interactions` (M3), `atoms-molecule` (M3), `atoms-chemistry` (M4),
`atoms-gpu` (M5), `atoms-render` (M2), `atoms-python` (M2), `atoms-viewer` (M2).
Each carries a `not yet implemented` notice, the milestone it targets, and an
explicit contract for what it must and must not do. A test
(`integration.rs::unimplemented_crates_declare_their_status`) fails if that
declaration is removed.

### 1.3 Documentation

| file | content |
|---|---|
| `docs/physics/conventions.md` | normative units, coordinates, quantum-number constraints, normalisation, `|psi|^2` vs `P(r)`, model limits |
| `docs/numerical-methods/quadrature-and-sampling.md` | methods, measured accuracy, truncation budgets, deliberate omissions |
| `docs/validation/reference-values.csv` | 733 lines, 732 generated reference values |
| `docs/validation/benchmarks-m1.md` | measured performance |
| `docs/validation/reference-cases.md` | (from M0) analytical targets |

---

## 2. Architecture decisions

1. **The physics core does not depend on graphics, GPU, or Python.**
   `atoms-core`, `atoms-numerics`, `atoms-orbitals` depend only on `serde`,
   `nalgebra`, `num-complex`, `rayon`, `rand`/`rand_chacha`. This is enforced by
   a test (`integration.rs::core_crates_have_no_graphics_gpu_or_python_dependencies`)
   that greps the manifests for `wgpu`, `pyo3`, `eframe`, `egui`, `cudarc`,
   `vulkan`, `ash` — so a future regression fails the build, not just review.

2. **Special functions live in `atoms-numerics`, not in the physics crates.**
   They are mathematics, not physics, and the quadrature used to validate them
   lives next to them.

3. **A validated-type boundary for quantum numbers.** `QuantumNumbers::new` is the
   only constructor, so an invalid triple cannot reach a kernel. Kernels take the
   validated type; the low-level `radial_unchecked` is `pub(crate)`.

4. **Probability density and radial probability density are separate APIs with
   separate units and separate documented meanings.** The distinction is the
   single most common error in this domain and is treated as a first-class API
   concern, not a comment.

5. **Errors are typed and specific**, hand-rolled (`QuantumNumberError`,
   `GridError`, `HarmonicError`, `DensityGridError`) rather than pulled from a
   derive macro, to keep the dependency footprint honest.

6. **Feature-gated heavy dependencies** (`pyo3`, `eframe`, `wgpu`) so that
   `cargo test --workspace` needs no Python toolchain, no GPU, and no display
   server.

7. **Deviations from the M0 layout:** M0 shipped a workspace list of 11 members.
   M1 added two (`tests/validation`, `benchmarks`) and split density evaluation
   from the wavefunction module. `tests/` at the top level is the
   `atoms-validation` crate rather than loose directories, because a virtual
   workspace root does not run unassociated test files.

8. **A bug in the M0-era draft was removed, not carried forward.**
   `EnergyLevel::verify_analytical` compared a formula against itself — a
   circular check that can never fail. It is gone; energy validation now runs
   against closed forms, the virial theorem, and the measured Lyman-alpha
   wavelength.

---

## 3. Tests executed and their actual results

```text
cargo test --workspace      ->  169 passed; 0 failed
cargo clippy --workspace --all-targets  ->  0 warnings
cargo fmt --all --check     ->  clean
```

| test binary | tests | result |
|---|---|---|
| `atoms-core` (lib) | 40 | ok |
| `atoms-numerics` (lib) | 22 | ok |
| `atoms-orbitals` (lib) | 63 | ok |
| `atoms-io` (lib) | 3 | ok |
| `atoms-validation` (lib) | 2 | ok |
| `atoms-validation` / `reference_values` | 5 | ok |
| `atoms-validation` / `physics` | 12 | ok |
| `atoms-validation` / `numerical` | 13 | ok |
| `atoms-validation` / `integration` | 9 | ok |
| **total** | **169** | **all pass** |

No test is `#[ignore]`d. No GPU test exists to skip (no GPU backend in M1).

### 3.1 Real bugs found by the tests during M1

These were found and fixed, not worked around:

1. **`<1/r^2>` was wrong by 9e-5 relative.** The integrand `r^{k+2} R(r)^2`
   tends to `R(0)^2` at `r = 0` for `k = -2`; the code substituted `0`, biasing
   the Simpson sum by `O(h)`. Regression test added
   (`density.rs::inverse_r_moments_handle_the_origin_correctly`).
2. **`reduced_mass` / wavelength comparison was initially wrong in the tests,
   which exposed a convention gap.** The closed form gives infinite-nuclear-mass
   energies; observed hydrogen lines are `~0.05 %` longer. Rather than loosen the
   tolerance, `reduced_mass_ratio` and
   `transition_wavelength_nm_with_reduced_mass` were added and both values are
   now asserted separately: `121.5023` nm (infinite mass) and `121.5684` nm
   (reduced mass, observed `121.567` nm).
3. **A finite-difference test produced a spurious `1.3e-6` residual on correct
   code** because `h = 1e-5` amplifies round-off by `1/h^2 = 1e10`. The step is
   now `4e-4 ~ eps^(1/4)` and the reason is documented in the test.
4. **A rejection-sampling test asserted impossible behaviour.** A too-small
   `pdf_max` cannot be detected by the sampler; the test now documents the
   bias quantitatively (variance `0.1254` vs `0.6569`).

### 3.2 Test-design corrections (my errors, not code errors)

* `zonal_harmonics_have_l_nodes` counted sign changes of `|Y|`, which can never
  change sign. Fixed to count the signed real function.
* `angular_probability_density_integrates_to_one` expected `1`; the correct value
  is `1/(2 pi)` because the `phi` integral supplies `2 pi`.
* 3-D grid normalisation was asserted for `n = 4`; the point count for a
  Cartesian grid grows roughly as `n^9`. Restricted to compact states, with the
  general case covered by marginal decomposition.
* Normalisation tolerances of `1e-9` were unreachable for `n = 5` at the chosen
  `r_max`; `r_max` is now derived from the analytic `<r>` and the measured tail
  is printed.

---

## 4. Scientific validation results

Every number below is *measured* in this workspace and printed by the test that
produced it.

### 4.1 Agreement with independent implementations

| quantity | reference | samples | measured max deviation |
|---|---|---|---|
| `Y_l^m(theta, phi)`, `l <= 5`, all `m` | `scipy.special.sph_harm` | 252 | `4.4e-16` absolute |
| `R_{nl}(r)`, `n <= 8` | `scipy.special.assoc_laguerre` | 288 | `4.7e-15` relative |
| `|psi_{nlm}|^2` | SciPy-derived | 108 | `1.4e-15` relative |
| `L_k^a(x)`, `k <= 39`, `a <= 61` | **exact rational arithmetic** | 84 | `1.0e-14` relative |

The SciPy reference is generated by `python/tests/generate_reference_values.py`
and committed as `docs/validation/reference-values.csv`. **Reference values are
generated, never transcribed by hand** — an earlier draft of a test contained
constants I had written from memory, which were wrong; that test was deleted and
replaced by the generator.

The `L_k^a` reference uses `fractions.Fraction`, so it carries *no*
floating-point error. This is what allowed the Laguerre recurrence to be cleared
of suspicion when a finite-difference test suggested otherwise: the recurrence is
accurate to `5e-16` at `k = 30, x = 40`, and the `2.6e-4` residual there was
truncation error in the *test method*.

### 4.2 Analytic results

| check | reference | measured |
|---|---|---|
| `E_n` for `n = 1..20` | `-1/(2n^2)` | `< 1e-16` |
| virial `E = -<1/r>/2` | analytic | `< 1e-16` |
| `l`-degeneracy of `E_n`, shell size `n^2` | analytic | exact |
| Lyman-alpha (infinite mass) | `121.5023` nm | `121.5023` nm |
| Lyman-alpha (reduced mass) | observed `121.567` nm | `121.5684` nm |
| Balmer-alpha (infinite / reduced mass) | `656.1123` / observed `656.46` | `656.1123` / `656.4696` nm |
| `<r>`, `<r^2>` | Kramers-Pasternack | `3.4e-13` / `2.4e-13` rel |
| `<1/r>`, `<1/r^2>` | `1/n^2`, `1/(n^3(l+1/2))` | `< 1e-7`, `< 1e-5` rel |
| addition theorem `sum_m |Y_l^m|^2 = (2l+1)/4pi`, `l <= 8` | analytic | `< 1e-12` |
| orthonormality of `Y_l^m`, `l <= 4` | analytic | `< 1e-6` |
| radial normalisation `int r^2 R^2 dr = 1`, `n <= 6` | analytic | `< 1e-8` |
| marginal normalisation `radial * angular * 2pi = 1`, `n <= 8` | analytic | `< 1e-8` |
| probability density non-negative everywhere on the grid | requirement | holds |

### 4.3 Nodal structure (the check that catches wrong conventions)

| structure | expected | measured |
|---|---|---|
| radial nodes of `R_{nl}` | `n - l - 1` | exact, `n <= 6`, all `l` |
| angular nodes of `Y_l^0` | `l` sign changes | exact, `l = 1..6` |
| `2p_{m=0}` in the `z = 0` plane | identically zero | `< 1e-30` |
| `2p_{|m|=1}` on the `z` axis | identically zero | `< 1e-30` |
| `2p_{|m|=1}` in the `z = 0` plane | **non-zero** | `> 1e-6` |
| `2p_{m=0}` on the `z` axis | **non-zero** | `> 1e-3` |

### 4.4 Differential-equation residuals

The implementations were checked against the equations that define them, not only
against other evaluations of the same closed form:

* `R_{nl}` satisfies the radial Schrodinger equation,
  `-1/2 R'' - R'/r + [l(l+1)/2r^2 - 1/r] R = -1/(2n^2) R`.
* `Y_l^m` is an eigenfunction of `L_z` with eigenvalue `m`.
* `L_+ Y_l^m = sqrt(l(l+1) - m(m+1)) Y_l^{m+1}` under the numerical raising
  operator.
* `P_l^m` and `L_k^a` satisfy their associated differential equations.

### 4.5 Sampling

| check | measured | reference |
|---|---|---|
| `P(r)` 1s, `<r>` | `1.50173` | `1.5` |
| `P(r)` 1s, `Var` | `0.75526` | `0.75` |
| `P(r)` 2p, `<r>` | `5.0` within `0.03` | `5.0` |
| angular `<cos^2 theta>` (5 cases) | agreement to `~5e-4` | Simpson quadrature |
| reproducibility, same seed | bit-identical | required |
| parallel evaluation | bit-identical | required |

---

## 5. Performance measurements

Full record in `docs/validation/benchmarks-m1.md`. Headline, measured on an
Apple M4 Max with 14 rayon threads:

| operation | measured |
|---|---|
| radial `R_nl(r)` | 122.0 M evals/s |
| spherical harmonic `Y_l^m` | 41.4 M evals/s |
| density on 3D grid (parallel) | 193–238 M points/s |
| density on 3D grid (1 thread) | 22.9–25.9 M points/s |
| measured parallel speedup | 7.9–10.4x on 14 threads |
| radial sampling (1 / 8 streams) | 21.5 / 160.7 M samples/s |

**No GPU numbers are reported**, because no GPU backend exists in M1. Speedup is
sub-linear in the thread count; the plausible causes are the M4 Max's mixed
performance/efficiency cores and memory traffic on the smaller grids, and the
measurements do not separate them. Nothing here is extrapolated to other
hardware.

---

## 6. Remaining limitations

Recorded honestly, not hidden:

**Physics**
* One electron, one nucleus. No screening, exchange, or correlation. `Z > 1` is
  the unscreened hydrogenic approximation, not a multi-electron atom.
* Infinite nuclear mass in the default energies. The reduced-mass factor is
  available and tested but not applied automatically.
* No spin, fine structure, Lamb shift, relativistic corrections, or finite
  nuclear size.
* No real (as opposed to complex) orbital forms such as `p_x`. Complex `Y_l^m`
  are the primary representation for M1; real combinations are deferred and are
  **not** silently substituted.

**Numerics**
* Laguerre accuracy beyond `n = 40` is not claimed and not tested.
* Transcribed radial integrals are limited to `k >= -2`; lower powers diverge.
* 3-D Cartesian normalisation is limited to compact states (point count grows as
  `n^9`); the general statement uses marginal decomposition.
* First-order-accurate grid sampling at cell corners, so 3-D integrals carry an
  `O(dx^2)` error (measured `4.4e-3` at `dx = 0.1875` bohr).

**Engineering**
* No CI configuration file yet. The commands are `cargo fmt --all --check`,
  `cargo clippy --workspace --all-targets`, `cargo test --workspace`; wiring them
  into a workflow is part of M2.
* `atoms-io` implements only the metadata JSON type. XYZ, NumPy array export,
  and a large-dataset container are deferred.
* The reference generator is run manually; the CSV is committed so the test suite
  does not depend on a Python environment.

**Explicitly not done in M1, as instructed**
* No molecular electronic-structure solver. No Hartree-Fock, DFT, or basis sets.
  No molecular property is computed anywhere in the workspace, and no
  quantum-chemistry claim is made.

---

## 7. Acceptance gates

| gate | status | evidence |
|---|---|---|
| All relevant unit, integration and validation tests pass | **pass** | 169/169, §3 |
| Full workspace test suite passes | **pass** | `cargo test --workspace`, exit 0 |
| Numerical tolerances and reference values recorded | **pass** | §4; `docs/physics/conventions.md`, `docs/numerical-methods/quadrature-and-sampling.md`, `reference-values.csv` |
| Benchmarks report actual measurements without unsupported claims | **pass** | §5; `docs/validation/benchmarks-m1.md` |
| Report lists changed files, tests, results, limitations, next milestone | **pass** | this document |
| `atoms-core` independent of GUI/render/GPU | **pass** | enforcing test, §2.1 |
| wgpu architecture (Vulkan/Metal) + optional CUDA preserved | **pass** | contracts in `atoms-render`, `atoms-gpu`; no GPU complexity added in M1 |
| No GPU complexity added without a justified interface | **pass** | `atoms-gpu` is a documented stub with no dependency on `wgpu` or `cudarc` in the default build |
| Rust and Python APIs documented and testable | **partial** | Rust API fully documented and tested. **Python API not implemented in M1** — it is the first item of M2; the binding crate is a documented stub with its contract fixed |
| No molecular solver or molecular-chemistry claim in M1 | **pass** | §6; `atoms-chemistry` and `atoms-molecule` contain no solver and are marked `M4`/`M3` |
| Upstream `kavan010/Atoms` unmodified | **pass** | the clone lives at `/tmp/atoms-ref` and was only read |

One gate is partial by design: the Python API is explicitly an M2 deliverable in
the milestone plan, so M1 cannot satisfy it. Flagged rather than glossed over.

---

## 8. Next milestone proposed: M2

**M2 — Python API and first 3D viewer.** Scoped as:

1. **PyO3 bindings** (`atoms-python`, feature `python`) exposing `Atom`,
   `orbital(n, l, m)`, `Grid3D`, `probability_density`, seeded sampling, and
   NumPy-compatible array transfer without gratuitous copies. Invalid parameters
   raise Python exceptions; nothing panics across the boundary.
2. **A reproducible Python example** that runs the documented target API and
   writes a cross-section image, with units stated explicitly in the API.
3. **`wgpu` renderer** (`atoms-render`) with WGSL shaders: isosurface and volume
   rendering of `DensityGrid`, configurable colour map and threshold, interactive
   camera. Backend and adapter reported at runtime.
4. **`atoms-viewer`** orbital explorer with quantum-number selection and 2D
   cross-sections, worker-thread computation so the render loop never blocks.
5. **CI workflow** running fmt, clippy, tests, and a conditional GPU test that
   skips cleanly with no adapter.
6. **Backend verification** on Metal (this machine) and a documented statement
   about Vulkan (tested or untested, explicitly).

Follow-on milestones unchanged: M3 interactions and molecular geometry,
M4 quantum chemistry, M5 GPU compute and advanced visualisation.

**Stop point.** M1 is complete and verified. I am not starting M2 without your
go-ahead.
