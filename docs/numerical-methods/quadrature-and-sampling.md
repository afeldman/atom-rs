# Numerical methods

This document records what numerical machinery exists in M1, why each piece was
chosen, its verified accuracy, and what was deliberately left out.

Guiding rule: **no numerical method is added until a concrete accuracy or
performance need has been demonstrated.** M1 therefore contains composite
Newton-Cotes quadrature and piecewise-linear inverse-CDF sampling, and nothing
more elaborate.

---

## 1. Special functions

### 1.1 Generalised Laguerre `L_k^a(x)`

Three-term recurrence, forward in `k`:

```text
L_0^a(x)   = 1
L_1^a(x)   = 1 + a - x
j L_j^a(x) = (2j - 1 + a - x) L_{j-1}^a(x) - (j - 1 + a) L_{j-2}^a(x)
```

**Accuracy.** Verified against a reference computed with **exact rational
arithmetic** (`fractions.Fraction`), which carries no floating-point error at
all. Across 84 evaluations spanning `k = 0..39`, `a = 1..61`, `x = 0.5..80`:

```text
max relative deviation = 1.0e-14   (worst case k=20, a=3, x=0.5)
```

The recurrence is accurate to a few ulp throughout the supported range. This
matters because the radial wavefunction is a product of this polynomial, a
power and an exponential, and cancellation inside the polynomial is the only
place where precision can be lost.

**Verified by:** `atoms-validation/tests/reference_values.rs`,
`laguerre_recurrence_matches_exact_rational_arithmetic`.

### 1.2 Associated Legendre `P_l^m(x)`

Upward recurrence in `l`, seeded at `m = m`, with the Condon-Shortley phase
accumulated explicitly:

```text
P_m^m(x)       = (-1)^m (2m-1)!! (1 - x^2)^{m/2}
P_{m+1}^m(x)   = x (2m+1) P_m^m(x)
(l-m) P_l^m(x) = x (2l-1) P_{l-1}^m(x) - (l+m-1) P_{l-2}^m(x)
```

The argument is clamped to `[-1, 1]` to absorb the round-off from `cos(theta)`.

**Accuracy.** Matches closed forms exactly for `l <= 3`; verified against the
associated Legendre differential equation for `l <= 5`; finite for
`l <= 30`, `m <= l`; and the resulting spherical harmonics agree with
`scipy.special.sph_harm` to `4.4e-16` absolute over 252 samples.

---

## 2. Quadrature

### 2.1 Composite Simpson's rule

`atoms_numerics::quadrature::simpson`. An odd interval count is bumped up to even
rather than panicking; `n < 2` is clamped to 2.

**Verified convergence.** On `int_0^1 e^x dx`:

| intervals | measured error ratio on halving `n` |
|---|---|
| 32 -> 64 | **16.00** |

The theoretically expected ratio for a fourth-order rule is 16, and the measured
value is 16.00. Exact for polynomials of degree `<= 3` (asserted directly).

### 2.2 Composite trapezoidal rule

Provided for cases where a lower-order rule is sufficient, and as the building
block of the CDF table in the sampler (where the piecewise-linear interpolation
matches the trapezoidal increment exactly, so the table and the inversion are
mutually consistent). Measured convergence ratio on halving `n`: `~4`, as
expected for second order.

### 2.3 Semi-infinite integration

`simpson_semi_infinite` applies the substitution `x = a + t/(1-t)` to map
`[a, inf)` onto `[0,1)`. Used to check that truncating radial integrals at a
finite `r_max` loses nothing.

### 2.4 Truncation of radial integrals

Rather than hide the truncation, the workspace measures it. For
`r_max = 8 <r> + 30` bohr with `<r>` the analytic expectation value:

| state | `<r>` (analytic) | `r_max` | `int_0^r_max P dr` | tail `P(r > r_max)` |
|---|---|---|---|---|
| 1s | 1.5 | 42 | 0.999999999999782 | 1.2e-33 |
| 3d | 10.5 | 114 | 0.999999999999610 | 2.9e-25 |
| 5s | 37.5 | 330 | 0.999999999999832 | 2.1e-41 |
| 6h | 39.0 | 342 | 0.999999999999794 | 3.5e-34 |
| 8s | 96.0 | 798 | 0.999999999999854 | 2.7e-60 |

Truncation error is below `1e-12`, i.e. far below the quadrature error, and the
tail itself is printed by the test so it is never merely assumed.

---

## 3. Sampling

`atoms_numerics::sampling`. Two strategies, both driven by an explicitly seeded
ChaCha8 stream (`SeededRng::from_seed`).

### 3.1 Inverse-CDF sampling (default)

`InverseCdf1D` tabulates the density on a uniform grid, normalises it, and
inverts by binary search with linear interpolation. The trapezoidal increments in
the table are the exact linear interpolant of the interpolation used to invert
it, so table and inversion are consistent by construction.

**Verified against analytic moments and against quadrature:**

| distribution | sampled | reference |
|---|---|---|
| `P(r)` for 1s, `<r>` (300 000 samples) | 1.50173 | 1.5 (exact) |
| `P(r)` for 1s, `Var(r)` | 0.75526 | 0.75 (exact) |
| `|P_1^0|^2`, `<cos^2 theta>` (400 000 samples) | 0.59908 | 0.60000 |
| `|P_1^1|^2`, `<cos^2 theta>` | 0.20015 | 0.20000 |
| `|P_2^0|^2`, `<cos^2 theta>` | 0.52357 | 0.52381 |
| `|P_2^2|^2`, `<cos^2 theta>` | 0.14284 | 0.14286 |
| `|P_3^1|^2`, `<cos^2 theta>` | 0.46612 | 0.46667 |

The angular reference is computed by Simpson quadrature — a different code path
from the sampler — so this is a genuine cross-check, not a self-comparison. The
agreement is at the level of the Monte-Carlo standard error
(`~sqrt(Var/N) ~ 0.002`).

### 3.2 Rejection sampling

Provided for cases where a bounded proposal is cheaper than building a table.

**Documented failure mode.** `pdf_max` must be a true upper bound on the density.
If it is too small the acceptance test becomes trivially true and the sampler
silently returns the *proposal* distribution. The function cannot detect this.
The test suite demonstrates the failure quantitatively: with a valid bound the
variance is `0.1254` against the analytic `0.125`; with an invalid bound it
inflates to `0.6569`, which is a detectable signature. `max_attempts == 0` is the
only "no sample" failure mode and returns `None`.

### 3.3 Reproducibility

Same seed -> identical sample sequence, on every run and every platform
(asserted for 5000 consecutive samples). Different seeds -> different sequences.
Parallel sampling uses one stream per chunk, seeded from the chunk index, so the
result is independent of thread scheduling.

---

## 4. Volumetric evaluation

`DensityGrid::probability_density` evaluates `|psi|^2` on a Cartesian grid with
`rayon`, one point per task, so the result is **bit-for-bit deterministic**
regardless of thread count (asserted by comparing two runs).

**Grid discretisation error.** For 3-D Riemann integration of a normalised
density:

| grid | deviation `|int|psi|^2 dV - 1|` |
|---|---|
| `+/-12` bohr, `dx = 0.1875` | 4.4e-3 |

This is a **grid** error, not a normalisation error: the integral is never
renormalised in the library, precisely so that the value can be used as a test.

**Why 3-D grid normalisation is limited to compact states.** A Cartesian grid
needs extent `~ n^2/Z` bohr and spacing fine enough to resolve `n-l-1` radial
nodes, so the point count grows roughly as `n^9`. Beyond `n = 2` this becomes
impractical (`n = 4` exceeds `10^9` points). The general normalisation statement
is therefore verified by *marginal decomposition*,

```text
int_0^inf r^2 R^2 dr  *  int_0^pi |Y|^2 sin(theta) d(theta)  *  2 pi  =  1
```

which is cheap, exact, and verified for `n <= 8` at `1e-8`.

---

## 5. Expectation values

`radial_expectation_value` integrates `r^{k+2} R(r)^2` by Simpson on
`[0, 15 n^2 + 40]` with 400 000 intervals.

**Validated against the closed forms** (`atoms_orbitals::expectations`):

| quantity | closed form | measured max relative deviation |
|---|---|---|
| `<r>` | `(3n^2 - l(l+1))/2` | 3.4e-13 |
| `<r^2>` | `n^2 (5n^2 + 1 - 3l(l+1))/2` | 2.4e-13 |
| `<1/r>` | `1/n^2` | `< 1e-7` |
| `<1/r^2>` | `1/(n^3 (l + 1/2))` | `< 1e-5` |

**Domain of validity.** `k >= -2`. The integrand is `r^{k+2} R(r)^2`, so the
`k = -2` case tends to `R(0)^2`, which is finite and non-zero for s orbitals.
Handling that limit correctly matters: substituting `0` for the endpoint value
biases the Simpson sum by `O(h)` and produced `<1/r^2> = 1.99982` instead of
`2.0` for the 1s state before the bug was fixed. It is now covered by a
regression test. `k < -2` diverges at the origin and returns `NaN`.

---

## 6. Step-size discipline in finite-difference tests

Tests that verify a differential equation by finite differences must choose `h`
deliberately. A central second difference has truncation error `O(h^2)` and
round-off error `O(eps/h^2)`, minimised at `h ~ eps^(1/4) ~ 4e-4` for `f64`.

This is not a detail. Using `h = 1e-5` amplifies round-off by `1/h^2 = 1e10` and
produced a spurious `1.3e-6` relative residual on a *correct* implementation.
The tests in this workspace either use the near-optimal step or measure the
residual against the orbital's maximum amplitude, and they say which and why.

Related: at high `k` the *same* test method fails for a different reason. The
derivatives of `L_k^a` grow steeply, so the finite-difference truncation error
in the Laguerre differential equation reaches `2.6e-4` at `k = 30`, while the
implementation is accurate to `5e-16` there (measured against exact rational
arithmetic). High-order accuracy is therefore established by the exact-arithmetic
comparison, not by differentiation.

---

## 7. Deliberately not implemented in M1

| method | why not |
|---|---|
| Gauss-Legendre quadrature | Simpson's rule already reaches machine precision on the integrands here; no measured need. |
| Adaptive quadrature | same. |
| Numerical ODE integration of the radial equation | the solution is analytic; integrating it numerically would introduce error without adding information. |
| Sparse or spectral grid representations | no demonstrated need; a dense Cartesian grid is what the renderer and the numerical tests both want. |
| GPU kernels | deferred to M5, and only for kernels that profiling shows to be bottlenecks. |
