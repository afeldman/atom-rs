# Physical conventions (normative)

This document is the single authoritative statement of the conventions used by
`atoms-rs`. Every public function documents which of these conventions it obeys;
where this document and a doc comment disagree, the doc comment is the bug.

---

## 1. Units

### 1.1 Internal representation: Hartree atomic units

**All internal computation is in Hartree atomic units.** In this system the
following constants are set to one by definition:

| quantity | atomic unit | symbol | value in SI (CODATA 2018) |
|---|---|---|---|
| length | bohr | `a0` | `5.29177210903e-11` m |
| energy | hartree | `Eh` | `4.3597447222071e-18` J |
| mass | electron mass | `me` | `9.1093837015e-31` kg |
| charge | elementary charge | `e` | `1.602176634e-19` C |
| action | reduced Planck constant | `hbar` | `1.054571817e-34` J s |
| time | `hbar/Eh` | — | `2.4188843265857e-17` s |

Consequences that the code relies on:

* the hydrogen Hamiltonian is `H = -1/2 laplacian - Z/r`;
* the ground-state energy is exactly `-0.5` (not `-13.6` anything);
* `1/(4 pi eps0) = 1`, so the Coulomb potential is `Z/r` with no prefactor;
* the speed of light is `137.035999084` (dimensionless).

### 1.2 Derived conversion factors

| conversion | factor |
|---|---|
| `1 hartree` | `27.211386245988` eV |
| `1 hartree` | `2625.4996394799` kJ/mol |
| `1 hartree` | `4.3597447222071e-18` J |
| `1 bohr` | `0.529177210903` Å |
| `1 u` (atomic mass unit) | `1822.888486209` electron masses |

These live in `atoms_core::units::constants` and
`atoms_core::units::conversions`. **They are the only place in the workspace
where a non-atomic unit appears.** Every other module works in atomic units and
says so.

### 1.3 Unit-aware API types

`LengthUnit` and `EnergyUnit` carry an explicit conversion method rather than a
bare scaling constant, so that a call site states which unit it is passing:

```rust
LengthUnit::Angstrom.to_bohr(1.0);        // 1.88972612462577
EnergyUnit::ElectronVolt.to_hartree(-13.6);
```

A parameter named `r` with no unit type is documented as bohr. There is no
function in this workspace that guesses a unit from the magnitude of its
argument.

---

## 2. Coordinate conventions

### 2.1 Spherical coordinates: physics convention

`(r, theta, phi)` with

* `r >= 0` — distance from the origin, bohr;
* `theta` — **polar (colatitude) angle measured from the `+z` axis**, `[0, pi]`;
* `phi` — **azimuthal angle measured in the `x`-`y` plane from the `+x` axis
  towards the `+y` axis**, `[0, 2 pi)`.

```text
x = r sin(theta) cos(phi)
y = r sin(theta) sin(phi)
z = r cos(theta)

r     = sqrt(x^2 + y^2 + z^2)
theta = acos(z / r)        (defined as 0 at r = 0)
phi   = atan2(y, x)        (normalised to [0, 2 pi))
```

**This is not the geography convention** (`theta` = azimuth, `phi` = polar) and
**not the engineering convention** (`theta` = azimuth from `+x`). It matches
`scipy.special.sph_harm` and standard quantum-mechanics texts.

The origin is handled explicitly: `theta` and `phi` are set to `0` rather than
left undefined, and every angular function in the workspace is either finite
there or multiplied by a vanishing radial factor.

### 2.2 Cartesian grids: buffer layout

Grids are C-order over `(i, j, k)` with **`i` varying fastest**:

```text
linear_index = (k * ny + j) * nx + i
```

Verified by `SpatialGrid3D::index` / `multi_index` round-trip tests and by a test
that the index map is unique and dense. The NumPy analogue is an array shaped
`(nz, ny, nx)` in C order; nothing in the workspace silently transposes data.

---

## 3. Quantum numbers and their constraints

| symbol | meaning | constraint |
|---|---|---|
| `n` | principal | `n >= 1` |
| `l` | orbital angular momentum | `0 <= l < n` |
| `m` | magnetic | `-l <= m <= l` |

Enforced in exactly one place, `QuantumNumbers::new`, which returns a typed error
naming the violated constraint. Every downstream kernel takes an already
validated `QuantumNumbers`, so an invalid triple cannot reach the numerics
through the typed API.

**Negative `m` is fully supported.** `m` is validated by magnitude
(`|m| <= l`), the classic unsigned-wraparound bug is covered by a test using
`i32::MIN`, and the complex phase for `m < 0` is pinned by tests against the
`Y_l^{-m} = (-1)^m conj(Y_l^m)` relation.

Node counts, which are testable consequences:

* radial nodes: `n - l - 1`
* angular nodes: `l`
* total: `n - 1`

---

## 4. Wavefunctions and their normalisation

### 4.1 Radial part

```text
rho       = 2 r / n
N_{nl}    = sqrt( (2/n)^3 (n-l-1)! / (2 n (n+l)!) )
R_{nl}(r) = N_{nl} exp(-rho/2) rho^l L_{n-l-1}^{2l+1}(rho)
```

with `L` the generalised Laguerre polynomial as defined in
`atoms_numerics::special::generalized_laguerre` (no binomial prefactor, not
normalised):

```text
L_0^a(x)   = 1
L_1^a(x)   = 1 + a - x
j L_j^a(x) = (2j - 1 + a - x) L_{j-1}^a(x) - (j - 1 + a) L_{j-2}^a(x)
```

Unit of `R`: `bohr^{-3/2}`.

### 4.2 Angular part

For `m >= 0`:

```text
Y_l^m(theta, phi) = N_{lm} P_l^m(cos theta) exp(i m phi)
N_{lm}            = sqrt( (2l+1)/(4 pi) * (l-m)!/(l+m)! )
```

with `P_l^m` **including** the Condon-Shortley phase `(-1)^m`.

For `m < 0`, with `M = -m`:

```text
Y_l^{-M} = (-1)^M conj( Y_l^M )
```

This is the SciPy convention, confirmed value-by-value against
`scipy.special.sph_harm` (see §6).

### 4.3 Full wavefunction

```text
psi_{nlm}(r, theta, phi) = R_{nl}(r) Y_l^m(theta, phi)
```

Normalised as

```text
int_{R^3} |psi|^2 dV = 1
```

`psi` is complex for `m != 0`. The complex phase is physical, not an artefact.

### 4.4 Probability density versus radial probability density

**These are different quantities with different units and different values.
The workspace keeps them strictly apart.**

| quantity | formula | meaning | unit | value at `r = 0` |
|---|---|---|---|---|
| probability density | `|psi|^2` | probability per unit volume | `bohr^{-3}` | `1/pi` for 1s; 0 for `l > 0` |
| radial probability density | `P(r) = r^2 |R_{nl}(r)|^2` | probability per unit *radius* | `bohr^{-1}` | **0 for every orbital** |
| angular probability density | `|Y_l^m(theta)|^2` | probability per unit solid angle | `sr^{-1}` | — |

Common errors this table is designed to prevent:

* `P(r)` is **not** `|psi|^2` sampled on a sphere;
* `int_0^inf |psi|^2 dr` is **not** 1, and `int_0^inf P(r) dr` is;
* `int_0^pi |Y|^2 sin theta dtheta = 1/(2 pi)`, **not** 1 — the missing factor is
  the `2 pi` from the azimuthal integral.

All three statements are asserted in the test suite.

---

## 5. Nuclear charge and the limits of the model

For a one-electron ion of charge `Z`:

```text
R_{nl}^{Z}(r) = Z^{3/2} R_{nl}(Z r)
E_n           = -Z^2 / (2 n^2)
```

which preserves normalisation and contracts the orbital by `1/Z`. Tested for
`Z = 1..4` against normalisation, `<r> = 1.5/Z` and the energy formula.

**Limits, stated plainly:**

* One electron, one nucleus. No screening, no electron-electron repulsion, no
  exchange, no correlation.
* Infinite nuclear mass. The finite-mass correction is a factor
  `mu/m_e = M/(M + m_e)`; for a proton this is `0.9994556794247658`. It is
  **not** applied automatically; `reduced_mass_ratio` gives the factor and
  `EnergyLevel::transition_wavelength_nm_with_reduced_mass` applies it. The
  infinite-mass convention is the one that makes `E_n = -1/(2n^2)` exact, and it
  predicts Lyman-alpha at `121.5023` nm rather than the observed `121.567` nm.
* No spin, no fine structure, no Lamb shift, no relativistic corrections, no
  finite nuclear size.
* Applying `Z > 1` to a species with more than one electron is the *unscreened*
  hydrogenic approximation, not a description of that atom.

---

## 6. Independent verification of these conventions

| what | how it was checked | measured result |
|---|---|---|
| `Y_l^m` values and phase | 252 samples against `scipy.special.sph_harm` | max abs deviation `4.4e-16` |
| `R_{nl}` values | 288 samples against `scipy.special.assoc_laguerre` | max rel deviation `4.7e-15` |
| `|psi|^2` | 108 samples against SciPy-derived reference | max rel deviation `1.4e-15` |
| `L_k^a` recurrence | 84 evaluations against exact rational arithmetic (`fractions.Fraction`) | max rel deviation `1.0e-14` |
| angular nodal structure | sign-change counting of `Y_l^0` | exactly `l` nodes, `l = 1..6` |
| radial nodal structure | sign-change counting of `R_{nl}` | exactly `n-l-1` nodes, `n <= 6` |
| coordinate round trip | `(r,theta,phi) -> (x,y,z) -> (r,theta,phi)` | `< 1e-14` absolute |
| addition theorem | `sum_m |Y_l^m|^2 = (2l+1)/(4 pi)`, `l <= 8` | `< 1e-12` |
| `<r>`, `<r^2>`, `<1/r>`, `<1/r^2>` | numerical quadrature vs Kramers-Pasternack closed forms | `< 1e-5` relative |

Reference data is generated, never transcribed: see
`python/tests/generate_reference_values.py` and
`docs/validation/reference-values.csv`.
