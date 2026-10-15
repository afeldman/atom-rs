# Python API reference

Package: `atoms` (extension module `atoms._atoms`, built with PyO3 and maturin).

This document is the authoritative statement of the Python-side contract. The
physics itself is documented in `docs/physics/conventions.md`.

## Installation (reproducible)

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install maturin pytest numpy scipy
cd crates/atoms-python
../../.venv/bin/maturin develop --release
```

`maturin develop` builds the Rust extension and installs it **editable** into the
active virtual environment, so edits to `python/atoms/*.py` take effect without
a rebuild. Rust changes require re-running it.

Verified working environment: CPython 3.15, maturin 1.15.0, numpy 2.5.4,
scipy 1.18.1, pyo3 0.30.1, on macOS/arm64. `pyproject.toml` declares
`requires-python = ">=3.9"` and `numpy>=1.22`, but only Python 3.15 has actually
been exercised; see "Platform verification" below.

## Conventions (normative)

### Units

**Every physical result is in Hartree atomic units.**

| quantity | unit |
|---|---|
| position, `r`, radii, grid coordinates | bohr |
| energy | hartree |
| probability density `|psi|^2` | bohr⁻³ |
| radial probability density `r^2|R|^2` | bohr⁻¹ |
| angular probability density `|Y_l^m|^2` | sr⁻¹ |

The `unit=` argument of `Grid3D` only affects how the grid's **own geometry** is
specified and reported (`extent`, `spacing`, `axes()`). It never changes the unit
of a computed physical quantity. `bohr_extent` and `bohr_spacing` are always
available for unit-free work.

Use `atoms.units.*` to convert results. Each function takes and returns a float;
vectorise with NumPy where needed.

### Angles

`theta` is the **polar angle measured from `+z`**, in `[0, pi]`. `phi` is the
**azimuth measured from `+x` towards `+y`**, in `[0, 2 pi)`. Radians. This is the
physics convention, matching `scipy.special.sph_harm_y(n, m, theta, phi)`.

### Complex wavefunctions

`Orbital.value_at` returns a Python `complex`. For `m != 0` the value is genuinely
complex; the imaginary part is a physical phase. `Orbital.probability_density`,
`probability_density_at` and `total_probability` return the real `|psi|^2`.

`psi_{n,l,-m} = (-1)^m conj(psi_{n,l,m})`. Negative `m` is fully supported and is
**not** the same as `|m|`.

### Arrays

| function | shape | dtype | order |
|---|---|---|---|
| `Orbital.probability_density(grid)` | `(nx, ny, nz)` | `float64` | C |
| `atoms.sample_radial(...)` | `(n,)` | `float64` | C |
| `atoms.sample_positions(...)` | `(n, 3)` | `float64` | C |
| `Grid3D.axes()` | three `(n_x,)`, `(n_y,)`, `(n_z,)` arrays | `float64` | C |

`out[i, j, k]` is the value at the grid point `(x_i, y_j, z_k)`. Therefore
`out[:, :, k]` is a plane of constant `z`, and
`out[i, j, k] * grid.volume_element_bohr3` is the probability in that cell.

**Axis order is a deliberate, tested decision.** The Rust grid stores values in
C order with `i` fastest (`(k*ny + j)*nx + i`); the NumPy array is built directly
in the NumPy convention with `k` fastest (`(i*ny + j)*nz + k`) so that no
transpose or extra copy is needed. Both layouts and their consistency are pinned
by tests: `crates/atoms-python/src/ordering.rs` (Rust, no Python needed) and
`python/tests/test_atoms.py::test_density_orientation_is_not_transposed`.

### Probability density vs radial probability density

`|psi|^2` and `P(r) = r^2|R_{nl}|^2` are different quantities with different
units. `P(0) = 0` for *every* orbital, including `s` orbitals, while
`|psi(0)|^2 != 0` for `s` orbitals. `int_0^inf P(r) dr = 1`, whereas
`int_0^pi |Y|^2 sin(theta) dtheta = 1/(2 pi)`.

### Errors

| situation | exception |
|---|---|
| `n == 0`, `l >= n`, `|m| > l` | `ValueError`, naming the violated constraint |
| unknown unit string | `ValueError` |
| grid with a resolution axis `< 2`, `lo >= hi`, non-finite extent | `ValueError` |
| `extent` not a 2-tuple or three pairs | `ValueError` |
| `k < -2` in `radial_expectation_value` (divergent) | `ValueError` |
| `n_samples == 0` | `ValueError` |
| profile function given a non-array, non-float | `TypeError`, naming the received type |
| `Orbital`/`Grid3D` passed where the other is expected | `TypeError` |
| `atoms.viewer.show(...)` | `NotImplementedError` (viewer is M2-C) |

No Rust panic crosses the FFI boundary.

## API

### `atoms.Atom`

| member | meaning |
|---|---|
| `Atom.hydrogen()` | `Z = 1`, proton mass, at the origin |
| `Atom.from_z(z)` | neutral species, `1 <= z <= 10`, else `ValueError` |
| `atom.orbital(n, l, m)` | a validated `Orbital` for that species' `Z` |
| `atom.symbol`, `atom.z` | IUPAC symbol, nuclear charge |
| `atom.mass_amu`, `atom.mass_electron_masses` | nuclear mass |
| `atom.nominal_configuration` | **descriptive metadata only**, never computed |
| `atom.hydrogenic_model_is_exact` | `True` only for hydrogen |
| `atom.position` | `(x, y, z)` in bohr |

### `atoms.Orbital`

| member | returns |
|---|---|
| `n`, `l`, `m`, `z` | quantum numbers and nuclear charge |
| `quantum_numbers()` | `(n, l, m)` |
| `designation()` | `"2p"`, `"4d (m=-1)"`, ... |
| `radial_nodes`, `angular_nodes`, `total_nodes` | `n-l-1`, `l`, `n-1` |
| `energy(unit="hartree")` | analytic `E_n = -Z^2/(2n^2)` |
| `radial_wavefunction(r)` | `R_{nl}(r)` in bohr^−3/2 |
| `radial_probability_density(r)` | `r^2 R^2` in bohr^−1 |
| `angular_probability_density(theta)` | `|Y_l^m(theta)|^2` in sr^−1 |
| `value_at(x, y, z)` | complex `psi` in bohr^−3/2 |
| `probability_density_at(x, y, z)` | `|psi|^2` in bohr^−3 |
| `probability_density(grid)` | `(nx, ny, nz)` `float64` array |
| `total_probability(grid)` | `sum |psi|^2 dV`, deliberately not renormalised |
| `expectation_r()`, `expectation_r_squared()` | analytic Kramers-Pasternack |
| `expectation_inverse_r()`, `expectation_inverse_r_squared()` | analytic |
| `radial_expectation_value(k)` | numerical `<r^k>`, `k >= -2` |
| `sampling_radius_max` | upper bound of the sampling domain, bohr |

The three profile functions accept **either a float or a 1-D `float64` ndarray**
and return the matching type: a float returns a float, an array returns a new
array of the same length. Passing a list raises `TypeError`; use
`np.asarray([...])` explicitly so that the conversion is visible.

### `atoms.Grid3D(extent, resolution, unit="bohr")`

* `extent` is `(lo, hi)` for a cube, or `((x0,x1),(y0,y1),(z0,z1))`.
* `resolution` is `(nx, ny, nz)`, each `>= 2`.
* The outermost samples lie exactly on the extent, so
  `spacing = (hi - lo) / (n - 1)`.
* Properties: `unit`, `resolution`, `shape`, `size`, `extent`, `bohr_extent`,
  `spacing`, `bohr_spacing`, `volume_element_bohr3`.
* `axes()` returns the three coordinate arrays in `unit`.

### Module functions

| function | meaning |
|---|---|
| `sample_radial(orbital, n_samples, seed=0)` | radii in bohr from `P(r)` |
| `sample_positions(orbital, n_samples, seed=0)` | `(n, 3)` Cartesian positions in bohr |
| `spherical_harmonic(l, m, theta, phi)` | complex `Y_l^m` |
| `spherical_harmonic_abs_squared(l, m, theta)` | `|Y_l^m|^2` |
| `energy_level(n, z, unit="hartree")` | analytic energy eigenvalue |
| `reduced_mass_ratio(M_in_electron_masses)` | `mu/m_e` |
| `metadata_json(...)` | reproducibility record as a JSON string |

Namespaces: `atoms.units`, `atoms.harmonics`, `atoms.sampling` (thin aliases of
the flat functions).

Sampling is a pure function of `seed`: the same seed and sample count reproduce
the same array exactly, on any platform. Radii come from the truncated radial
density on `[0, 8<r> + 30]`; the neglected tail is below `1e-20` for the states
tested (`crates/atoms-orbitals/src/sampling.rs`).

## Scope and limitations

* One electron, unscreened, infinite nuclear mass. `Z > 1` is the hydrogenic
  approximation, not a multi-electron atom.
* No molecular electronic structure: no Hartree-Fock, no DFT, no basis sets, and
  no molecular property of any kind.
* Complex harmonics are the primary representation. Real orbitals such as `p_x`
  are **not** provided and are not silently substituted.
* `atoms.viewer` exists but raises `NotImplementedError`; the viewer is M2-C.
* No type stubs (`.pyi`) ship yet. `py.typed` is present as a placeholder;
  docstrings are complete, so `help(atoms.Orbital)` is accurate.

## Platform verification

| item | status |
|---|---|
| CPython 3.15 on macOS/arm64 | **tested**: 53 pytest tests pass, example runs |
| wheel built | `atoms-0.1.0-cp315-cp315-macosx_11_0_arm64.whl` |
| CPython 3.9–3.14 | **untested** (declared supported, never exercised) |
| Linux / Windows | **untested** |

A platform is only listed as tested if the test suite was actually run there.
