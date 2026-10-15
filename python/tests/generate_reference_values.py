#!/usr/bin/env python3
"""Generate independent reference values for hydrogenic quantities.

This script produces ``docs/validation/reference-values.csv`` from an
**independent implementation**: SciPy's ``special.sph_harm``,
``special.assoc_laguerre`` and ``special.gamma``. The Rust test suite reads the
resulting CSV and compares it against the Rust kernels, so the two
implementations are developed and evaluated separately.

The generator is deterministic: no random numbers, no seeds. Running it twice
must produce byte-identical output.

Usage::

    python generate_reference_values.py [output_path]

Requires: numpy, scipy.
"""

from __future__ import annotations

import csv
import sys
from fractions import Fraction
from math import factorial
from pathlib import Path

import numpy as np
from scipy import special

# ---------------------------------------------------------------------------
# Reference implementation (independent of the Rust code under test)
# ---------------------------------------------------------------------------


def radial_wavefunction(n: int, l: int, r: np.ndarray | float) -> np.ndarray:
    """R_{nl}(r) in atomic units, via scipy.special.assoc_laguerre.

    R = sqrt((2/n)^3 (n-l-1)! / (2 n (n+l)!))
        * exp(-rho/2) * rho^l * L_{n-l-1}^{2l+1}(rho),  rho = 2 r / n
    """
    rho = 2.0 * np.asarray(r, dtype=float) / n
    norm = np.sqrt(
        (2.0 / n) ** 3
        * special.gamma(n - l)
        / (2.0 * n * special.gamma(n + l + 1))
    )
    lag = special.assoc_laguerre(rho, n - l - 1, 2 * l + 1)
    return norm * np.exp(-rho / 2.0) * rho**l * lag


def spherical_harmonic(l: int, m: int, theta: float, phi: float) -> complex:
    """Y_l^m(theta, phi) with theta = polar angle from +z, phi = azimuth from +x.

    SciPy renamed and re-ordered this function: ``sph_harm_y(n, m, theta, phi)``
    is the current spelling (polar angle first), replacing the confusing
    ``sph_harm(m, n, theta_azimuthal, phi_polar)``. Both are handled so that the
    generator runs against either SciPy generation.
    """
    if hasattr(special, "sph_harm_y"):
        return complex(special.sph_harm_y(l, m, theta, phi))
    # Pre-1.15 SciPy: signature is (m, n, theta_azimuthal, phi_polar).
    return complex(special.sph_harm(m, l, phi, theta))


def psi(n: int, l: int, m: int, r: float, theta: float, phi: float) -> complex:
    return float(radial_wavefunction(n, l, r)) * spherical_harmonic(l, m, theta, phi)


# ---------------------------------------------------------------------------
# Value sets
# ---------------------------------------------------------------------------

# Angles chosen to avoid the exactly-nodal points where a relative comparison
# would be meaningless, while still covering every octant.
ANGLES = [
    (0.0, 0.0),
    (0.35, 1.20),
    (1.10, 0.60),
    (1.5707963267948966, 2.40),
    (2.20, 5.50),
    (2.90, 3.30),
    (3.141592653589793, 0.0),
]

RADII = [0.0, 0.05, 0.25, 1.0, 2.5, 5.0, 11.0, 25.0]


def write_harmonics(writer: csv.writer) -> int:
    """Columns: kind, l, m, theta, phi, re, im, unused."""
    count = 0
    for l in range(0, 6):
        for m in range(-l, l + 1):
            for theta, phi in ANGLES:
                y = spherical_harmonic(l, m, theta, phi)
                writer.writerow(
                    [
                        "harmonic",
                        l,
                        m,
                        repr(theta),
                        repr(phi),
                        repr(y.real),
                        repr(y.imag),
                        "",
                    ]
                )
                count += 1
    return count


def write_radial(writer: csv.writer) -> int:
    """Columns: kind, n, l, r, unused, R(r), unused, unused."""
    count = 0
    for n in range(1, 9):
        for l in range(0, n):
            for r in RADII:
                v = float(radial_wavefunction(n, l, r))
                writer.writerow(["radial", n, l, repr(r), "", repr(v), "", ""])
                count += 1
    return count


def write_density(writer: csv.writer) -> int:
    """Columns: kind, n, l, m, r, theta, phi, |psi|^2."""
    count = 0
    for n, l, m in [
        (1, 0, 0),
        (2, 0, 0),
        (2, 1, 0),
        (2, 1, 1),
        (2, 1, -1),
        (3, 2, 0),
        (3, 2, -2),
        (4, 3, 3),
        (5, 4, -1),
    ]:
        for r in [0.2, 1.0, 3.5, 9.0]:
            for theta, phi in [(0.4, 0.9), (1.2, 3.0), (2.7, 5.2)]:
                value = abs(psi(n, l, m, r, theta, phi)) ** 2
                writer.writerow(
                    [
                        "density",
                        n,
                        l,
                        m,
                        repr(r),
                        repr(theta),
                        repr(phi),
                        repr(value),
                    ]
                )
                count += 1
    return count


def laguerre_exact(k: int, alpha: int, x: float):
    """Exact L_k^alpha(x) using rational arithmetic.

    ``L_k^a(x) = sum_{i=0}^{k} (-1)^i C(k+a, k-i) x^i / i!``

    Because the coefficients are rational and the input ``x`` is represented as
    an exact fraction, this evaluation carries **no floating-point error** and
    therefore serves as an independent accuracy reference for the Rust
    recurrence (which is evaluated in f64).

    Returns a ``fractions.Fraction``; callers convert to float explicitly.
    """
    xr = Fraction(str(x))
    total = Fraction(0)
    m = k + alpha
    for i in range(0, k + 1):
        binom = Fraction(factorial(m), factorial(k - i) * factorial(m - (k - i)))
        total += Fraction((-1) ** i) * binom * xr**i / Fraction(factorial(i))
    return total


def write_laguerre(writer: csv.writer) -> int:
    """Columns: kind, k, alpha, x, unused, exact value as float, unused, unused.

    These values are computed with exact rational arithmetic
    (``fractions.Fraction``) and only converted to ``float`` at the very end, so
    the reference itself is error-free to within the final rounding.
    """
    count = 0
    cases = [
        (0, 1),
        (1, 1),
        (2, 1),
        (5, 1),
        (10, 1),
        (20, 1),
        (30, 1),
        (39, 1),
        (10, 3),
        (20, 3),
        (10, 41),
        (20, 41),
        (10, 61),
        (15, 7),
    ]
    for k, alpha in cases:
        for x in [0.5, 2.0, 5.0, 20.0, 40.0, 80.0]:
            value = float(laguerre_exact(k, alpha, x))
            writer.writerow(["laguerre", k, alpha, repr(x), "", repr(value), "", ""])
            count += 1
    return count


def main() -> int:
    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parents[2]
        / "docs"
        / "validation"
        / "reference-values.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "a", "b", "c", "d", "e", "f", "g"])
        nh = write_harmonics(w)
        nr = write_radial(w)
        nd = write_density(w)
        nl = write_laguerre(w)

    print(f"wrote {out}")
    print(f"  harmonics : {nh}")
    print(f"  radial    : {nr}")
    print(f"  density   : {nd}")
    print(f"  laguerre  : {nl} (exact rational arithmetic)")
    print(f"  numpy {np.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
