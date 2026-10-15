# M2-A — Python bindings and NumPy array transfer: completion report

**Status: complete, pending review of the API semantics in §8.**
Implementation commit: `421c19f`. This report is committed separately so that
the implementation commit stays exactly as reported.

---

## 1. Exact commit and working-tree state

```text
$ git log --oneline -1
421c19f M2-A: Python bindings with NumPy array transfer

$ git status --short | wc -l
0
```

Working tree clean at the start of this report. No build artifact is tracked:
the compiled extension (`crates/atoms-python/python/atoms/_atoms.cpython-315-darwin.so`)
is ignored, along with `.venv/`, `__pycache__/` and generated example output.

## 2. Files changed (22)

| file | change |
|---|---|
| `crates/atoms-python/src/bindings.rs` | new — the PyO3 extension module (no physics) |
| `crates/atoms-python/src/ordering.rs` | new — NumPy axis-order helpers, Rust-tested without Python |
| `crates/atoms-python/src/lib.rs` | rewritten — module docs and feature gating |
| `crates/atoms-python/Cargo.toml` | pyo3/numpy 0.30 behind the `python` feature |
| `crates/atoms-python/pyproject.toml` | new — maturin, mixed Rust/Python layout |
| `crates/atoms-python/python/atoms/__init__.py` | new — package docstring, namespaces, viewer stub |
| `crates/atoms-python/python/atoms/py.typed` | new |
| `crates/atoms-python/README.md` | new |
| `crates/atoms-orbitals/src/sampling.rs` | new — `OrbitalSampler`, 9 tests |
| `crates/atoms-orbitals/src/lib.rs` | export the sampling module |
| `python/tests/test_atoms.py` | new — API contract, 40 tests |
| `python/tests/test_parity.py` | new — numerical parity, 13 tests |
| `python/tests/generate_reference_values.py` | handle `sph_harm_y` and the pre-rename `sph_harm` |
| `python/examples/orbital_explorer.py` | new — reproducible example, dependency-free PNG writer |
| `docs/python-api.md` | new — the Python contract |
| `docs/validation/reference-values.csv` | regenerated with the current SciPy |
| `tests/validation/tests/reference_values.rs` | provenance updated, cross-generation comparison recorded |
| `Cargo.toml`, `Cargo.lock`, `README.md`, `.gitignore` | dependency versions, status, artifacts |

## 3. Commands executed

```bash
python3 -m venv .venv
.venv/bin/python -m pip install maturin pytest numpy scipy
cd crates/atoms-python && ../../.venv/bin/maturin develop --release
cargo test --workspace
cargo clippy --workspace --all-targets
cargo fmt --all --check
.venv/bin/python -m pytest python/tests -q
.venv/bin/python python/examples/orbital_explorer.py
.venv/bin/python python/tests/generate_reference_values.py
```

Environment: CPython 3.15.0, maturin 1.15.0, pyo3 0.30.1, numpy 2.5.4,
scipy 1.18.1, Rust 1.98.1, macOS/arm64.

## 4. Test results

```text
cargo test --workspace                   ->  183 passed; 0 failed
cargo clippy --workspace --all-targets   ->    0 warnings
cargo fmt --all --check                  ->  clean
.venv/bin/python -m pytest python/tests  ->   53 passed; 0 failed
```

| suite | before M2-A | after |
|---|---|---|
| Rust workspace | 169 | **183** (+9 sampling, +5 array ordering) |
| Python (pytest) | — | **53** |

No test is skipped or `#[ignore]`d except the three SciPy-dependent parity tests,
which skip with an explicit reason only when SciPy is absent (it is present here,
so all 53 ran).

### 4.1 Numerical parity

| comparison | samples | measured deviation |
|---|---|---|
| `R_{nl}` vs the Rust-validated reference CSV | 288 | 4.65e-15 rel |
| `Y_l^m` vs the reference CSV | 252 | 1.22e-15 abs |
| `\|psi\|^2` vs the reference CSV | 108 | 6.81e-15 rel |
| `Y_l^m` vs `scipy.special.sph_harm_y` directly | 252 | 1.22e-15 abs |
| `R_{nl}` vs `scipy.special.assoc_laguerre` directly | 108 | 2.98e-15 rel |

Plus, through the public Python surface only: harmonic orthonormality
(`< 1e-8`), the addition theorem (`< 1e-12`, `l <= 6`), angular normalisation
`int |Y|^2 sin(theta) dtheta = 1/(2 pi)`, radial normalisation
`int P(r) dr = 1`, and sampler moments (sampled `<r> = 1.5046` vs analytic `1.5`
for 1s).

### 4.2 Reference-data regeneration

Regenerating with numpy 2.5.4 / scipy 1.18.1 against the previously committed
values (numpy 2.0.2 / scipy 1.13.1):

| kind | rows | max absolute difference |
|---|---|---|
| harmonic | 252 | 9.99e-16 |
| radial | 288 | 0 (bit-identical) |
| density | 108 | 3.80e-19 |
| laguerre (exact rational) | 84 | 0 (bit-identical) |

A SciPy generation change therefore moves the reference only at the
floating-point round-off level, well below every test tolerance.

### 4.3 Errors found and fixed during M2-A

All four were mine, in tests, not in the implementation:

1. **A test point where `m = +1` and `m = -1` coincide.** I sampled `phi = pi/2`,
   where `Y_1^{1}` and `Y_1^{-1}` are equal by construction, so the test could not
   distinguish them. Replaced by the actual convention,
   `psi_{n,l,-m} = (-1)^m conj(psi_{n,l,m})`, checked at general points.
2. **A wrong 2p_z maximum location.** I asserted the density peak sits at a grid
   extreme; it sits at `z = +-2` bohr, the analytic maximum of `z^2 e^{-|z|}`.
   The test now asserts the physically correct position, which is a stronger
   statement than the one it replaced.
3. **A property called as a method** (`g.bohr_extent()`); it is a `#[getter]`.
4. **A too-tight summation tolerance** (`1e-14`) between two sums accumulated in
   different orders. Now the test asserts bit-for-bit equality of the values
   (after layout mapping) *and* states a separate `1e-12` tolerance for the sums,
   with the reason.

Also corrected: `np.trapezoid` over a non-endpoint array silently omits the last
interval, which showed up as a `0.6 %` normalisation deficit in one test.

## 5. Platform and backend verification

| item | status | evidence |
|---|---|---|
| CPython 3.15 on macOS/arm64 | **tested** | 53/53 pytest pass; example runs |
| wheel build | **tested** | `atoms-0.1.0-cp315-cp315-macosx_11_0_arm64.whl` |
| editable install | **tested** | `pip show atoms` -> version 0.1.0, module resolves into the repo |
| CPython 3.9–3.14 | **untested** | declared via `requires-python`, never exercised |
| Linux / Windows | **untested** | not available here |
| GPU / CUDA backend | **not applicable** | no GPU backend exists in M2-A |

**No backend is reported as verified because its code compiles.** The GPU
backends belong to M2-B, M2-C and M2-D and are untouched here.

## 6. Python installation results

```text
$ .venv/bin/pip show atoms
Name: atoms
Version: 0.1.0
Summary: Hydrogen-like quantum orbitals, densities and sampling (Rust core, Python API)
```

`import atoms` resolves to
`crates/atoms-python/python/atoms/__init__.py` (editable), and the compiled
module `atoms._atoms` is loaded from the same directory. `atoms.__version__` is
`0.1.0`, taken from `CARGO_PKG_VERSION` so it cannot drift from the crate.

`maturin develop --release` reported: *Built wheel for CPython 3.15 ...
Setting installed package as editable ... Installed atoms-0.1.0*.

## 7. Benchmark measurements (new in M2-A)

Measured from Python, median of 7 runs, on the M4 Max:

| operation | workload | median | derived throughput |
|---|---|---|---|
| `Orbital.probability_density` | 226 981 points (61³) | 1.32 ms | 171.5 M pts/s |
| `Orbital.probability_density` | 912 673 points (97³) | 2.67 ms | 341.8 M pts/s |
| `Orbital.probability_density` | 2 803 221 points (141³) | 8.48 ms | 330.6 M pts/s |
| `atoms.sample_positions` | 100 000 samples | 13.71 ms | 7.3 M samples/s |
| `atoms.sample_positions` | 1 000 000 samples | 173.56 ms | 5.8 M samples/s |
| `Orbital.radial_wavefunction` | 100 000-point array | 0.98 ms | 101.5 M evals/s |

**On "avoid unnecessary copies".** The argument is measured, not asserted:

* `probability_density` reaches 330.6 M points/s *through the Python boundary*,
  which is at or above the 238 M points/s the pure-Rust harness measured for the
  same operation. A copy of a 2.8 M-element array would cost roughly 22 MB of
  traffic; the observed throughput leaves no room for one.
* `radial_wavefunction` on a 100 000-element input runs at 101.5 M evals/s,
  against 122 M evals/s from the Rust harness. The difference is the array
  allocation for the result, not per-element Python overhead — a per-element cost
  would be three orders of magnitude larger.
* For reference, `arr.sum()` on the 2.8 M-element returned array takes 0.58 ms,
  which is the memory-bandwidth floor for touching it once. The array is a plain
  contiguous `float64` NumPy array.

**`sample_positions` is slower than the Rust sampling figure (5.8 vs 21.5 M/s)
and that is expected, not a defect:** the Rust benchmark measures radial sampling
only, whereas `sample_positions` draws a radius, a polar angle and an azimuth per
point and applies trigonometry, returning `(n, 3)` Cartesian coordinates. No
claim is made that the two numbers are comparable.

## 8. Decisions that need your sign-off

The instruction was to define the API semantics and validate the units **before**
exposing them publicly, so these are stated explicitly rather than buried:

1. **Units policy.** All physical results are in atomic units, always. `unit=`
   on `Grid3D` affects only the grid's own geometry, never the unit of a computed
   density. The alternative (returning `bohr^-3` or `angstrom^-3` depending on the
   grid's unit) was rejected because it silently changes the meaning of a number
   based on a distant argument.
2. **Axis order.** `out[i, j, k]` is the value at `(x_i, y_j, z_k)`, shape
   `(nx, ny, nz)`, C order. This is the NumPy-natural convention and makes
   `out[:, :, k]` a constant-`z` plane, but it is the *reverse* of the Rust
   buffer's internal order. The array is built directly in the NumPy order so no
   transpose is needed. If you would rather expose the Rust order, now is the
   time to say so.
3. **Profile functions: scalar or array.** `radial_wavefunction`,
   `radial_probability_density` and `angular_probability_density` accept a float
   or a 1-D `float64` array and return the matching type. A Python list raises
   `TypeError` rather than being converted implicitly.
4. **`atoms.viewer.show` exists but raises `NotImplementedError`.** The signature
   is frozen so the intended API is visible, and it fails loudly. If you would
   prefer the name to be absent until M2-C, that is a one-line change.
5. **Sampling is in Rust, not Python.** `OrbitalSampler` lives in
   `atoms-orbitals` and is tested there; the binding is a thin loop. This keeps
   the reproducibility guarantee (`seed -> identical samples`) enforceable by the
   Rust test suite rather than only by pytest.

## 9. Viewer capabilities

**None.** No renderer, no window, no image output from the viewer. What exists:

* `atoms.viewer.show(orbital=..., quantity=...)` raises `NotImplementedError`
  with a message pointing at M2-C.
* The example `python/examples/orbital_explorer.py` produces 2D cross-section
  PNGs (141×141) of the 3D density using a dependency-free PNG writer, a
  log-scaled colour map, a metadata record and a summary. These are static
  images, not an interactive viewer, and are described as such.

## 10. Remaining limitations

* **No GPU rendering and no viewer.** M2-B and M2-C are untouched.
* Python support is verified on one interpreter (3.15) and one platform
  (macOS/arm64). `requires-python = ">=3.9"` is a declaration, not a test result.
* No `.pyi` type stubs. `py.typed` is present as a placeholder; docstrings are
  complete, so `help(atoms.Orbital)` is accurate, but static type checkers get
  no signal from the compiled module.
* The extension must be rebuilt with `maturin develop` after any Rust change;
  Python-only changes are picked up immediately by the editable install.
* Sampling of high-`n` states allocates a larger CDF table (`20 000 + 2 000 n`
  radial points). No memory ceiling is enforced; a caller sampling many high-`n`
  orbitals could accumulate tables.
* Preserved M1 limitations stand unchanged: one electron, unscreened, infinite
  nuclear mass by default, no spin or relativistic corrections, no real orbitals.
* No CI workflow yet (M2-D). The commands above are the intended CI steps.

## 11. Next: M2-B

Portable `wgpu` rendering foundation, in this order:

1. `atoms-render`: device/adapter initialisation with graceful failure, backend
   and adapter reported at runtime.
2. WGSL shaders; isosurface and volume rendering of the density data the Python
   and Rust APIs already produce.
3. Validation of rendered data against the CPU reference — the same values the
   M1 tests already pin, so the renderer is checked against known physics rather
   than against a picture that merely looks right.
4. A CPU fallback so every scientific calculation is available without a GPU.
5. Explicit statement of what is *not* claimed: no GPU-accelerated scientific
   computation until real compute kernels exist and are benchmarked.

I will proceed to M2-B on your word, or directly if you prefer — the API
decisions in §8 are the only thing I would rather confirm first, because the
viewer's data path is built on top of them.
