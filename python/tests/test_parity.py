"""Numerical parity of the Python API.

Two independent references are used:

1. `docs/validation/reference-values.csv`, the committed data file that the
   **Rust** test suite also validates against SciPy and against exact rational
   arithmetic. Agreement here is Python-vs-Rust parity on identical inputs.
2. SciPy directly (`scipy.special.sph_harm_y`, `scipy.special.assoc_laguerre`),
   which is an implementation the Rust code was developed independently of.

If SciPy is not importable, the direct comparison is skipped with an explicit
reason rather than silently passing.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

import atoms

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CSV = REPO_ROOT / "docs" / "validation" / "reference-values.csv"

try:
    from scipy import special as sp

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_SCIPY = False

requires_scipy = pytest.mark.skipif(
    not HAVE_SCIPY, reason="scipy is not installed; direct cross-check skipped"
)


def load_rows(kind: str) -> list[list[str]]:
    assert REFERENCE_CSV.is_file(), (
        f"{REFERENCE_CSV} is missing; regenerate it with "
        "python python/tests/generate_reference_values.py"
    )
    rows = []
    with REFERENCE_CSV.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
        assert header == ["kind", "a", "b", "c", "d", "e", "f", "g"], header
        for row in reader:
            if row and row[0] == kind:
                rows.append(row)
    return rows


def spherical_to_cartesian(r: float, theta: float, phi: float) -> tuple[float, float, float]:
    st = math.sin(theta)
    return (r * st * math.cos(phi), r * st * math.sin(phi), r * math.cos(theta))


# ---------------------------------------------------------------------------
# Parity with the Rust-validated reference data
# ---------------------------------------------------------------------------


def test_reference_file_has_the_expected_size():
    total = sum(len(load_rows(k)) for k in ("harmonic", "radial", "density", "laguerre"))
    assert total == 732, total


def test_radial_wavefunction_parity_with_rust_reference():
    rows = load_rows("radial")
    assert len(rows) == 288
    h = atoms.Atom.hydrogen()
    worst = 0.0
    worst_ctx = ""
    for row in rows:
        n, l, r, value = int(row[1]), int(row[2]), float(row[3]), float(row[5])
        got = h.orbital(n, l, 0).radial_wavefunction(r)
        rel = abs(got - value) / abs(value) if value != 0.0 else abs(got)
        if rel > worst:
            worst, worst_ctx = rel, f"n={n} l={l} r={r}"
        assert rel < 1e-12, f"n={n} l={l} r={r}: python={got} rust={value}"
    print(f"max relative R_nl deviation vs Rust reference: {worst:.3e} ({worst_ctx})")


def test_spherical_harmonic_parity_with_rust_reference():
    rows = load_rows("harmonic")
    assert len(rows) == 252
    worst = 0.0
    worst_ctx = ""
    for row in rows:
        l, m, theta, phi = int(row[1]), int(row[2]), float(row[3]), float(row[4])
        re, im = float(row[5]), float(row[6])
        got = atoms.harmonics.value(l, m, theta, phi)
        dev = abs(got - complex(re, im))
        if dev > worst:
            worst, worst_ctx = dev, f"l={l} m={m} theta={theta} phi={phi}"
        assert dev < 1e-13, f"l={l} m={m}: python={got} rust={complex(re, im)}"
    print(f"max |Y_python - Y_rust|: {worst:.3e} ({worst_ctx})")


def test_probability_density_parity_with_rust_reference():
    rows = load_rows("density")
    assert len(rows) == 108
    h = atoms.Atom.hydrogen()
    worst = 0.0
    worst_ctx = ""
    for row in rows:
        n, l, m = int(row[1]), int(row[2]), int(row[3])
        r, theta, phi = float(row[4]), float(row[5]), float(row[6])
        expected = float(row[7])
        x, y, z = spherical_to_cartesian(r, theta, phi)
        got = h.orbital(n, l, m).probability_density_at(x, y, z)
        rel = abs(got - expected) / abs(expected)
        if rel > worst:
            worst, worst_ctx = rel, f"n={n} l={l} m={m} r={r}"
        assert rel < 1e-11, f"n={n} l={l} m={m}: python={got} rust={expected}"
    print(f"max relative |psi|^2 deviation vs Rust reference: {worst:.3e} ({worst_ctx})")


def test_negative_m_is_covered_by_the_reference_data():
    rows = load_rows("density")
    assert any(int(r[3]) < 0 for r in rows), "no negative m in the density reference"
    hrows = load_rows("harmonic")
    assert any(int(r[2]) < 0 for r in hrows), "no negative m in the harmonic reference"


# ---------------------------------------------------------------------------
# Direct comparison against SciPy (independent implementation)
# ---------------------------------------------------------------------------


@requires_scipy
def test_scipy_spherical_harmonic_convention_is_pinned():
    """Pin the SciPy argument order before relying on it.

    `scipy.special.sph_harm_y(n, m, theta, phi)` takes the polar angle first.
    This asserts that reading against the closed form Y_1^0 = sqrt(3/4pi) cos(theta)
    so that a later convention change is diagnosed here rather than producing
    confusing mismatches downstream.
    """
    theta = 0.9
    phi = 0.4
    expected = math.sqrt(3.0 / (4.0 * math.pi)) * math.cos(theta)
    assert abs(sp.sph_harm_y(1, 0, theta, phi) - expected) < 1e-14


@requires_scipy
def test_harmonics_match_scipy_directly():
    worst = 0.0
    worst_ctx = ""
    for l in range(0, 6):
        for m in range(-l, l + 1):
            for theta in (0.0, 0.35, 1.1, 1.5707963267948966, 2.2, 2.9, math.pi):
                for phi in (0.0, 0.6, 2.4):
                    expected = complex(sp.sph_harm_y(l, m, theta, phi))
                    got = atoms.harmonics.value(l, m, theta, phi)
                    dev = abs(got - expected)
                    if dev > worst:
                        worst, worst_ctx = dev, f"l={l} m={m} theta={theta} phi={phi}"
                    assert dev < 1e-12, f"l={l} m={m} theta={theta} phi={phi}: {got} vs {expected}"
    print(f"max |Y_atoms - Y_scipy| over 252 evaluations: {worst:.3e} ({worst_ctx})")


@requires_scipy
def test_radial_wavefunction_matches_scipy_directly():
    h = atoms.Atom.hydrogen()
    worst = 0.0
    worst_ctx = ""
    for n in range(1, 7):
        for l in range(n):
            for r in (0.0, 0.1, 0.75, 2.5, 7.0, 20.0):
                rho = 2.0 * r / n
                norm = math.sqrt(
                    (2.0 / n) ** 3
                    * sp.gamma(n - l)
                    / (2.0 * n * sp.gamma(n + l + 1))
                )
                expected = float(norm * np.exp(-rho / 2.0) * rho**l
                                 * sp.assoc_laguerre(rho, n - l - 1, 2 * l + 1))
                got = h.orbital(n, l, 0).radial_wavefunction(r)
                rel = abs(got - expected) / abs(expected) if expected != 0.0 else abs(got)
                if rel > worst:
                    worst, worst_ctx = rel, f"n={n} l={l} r={r}"
                assert rel < 1e-12, f"n={n} l={l} r={r}: {got} vs {expected}"
    print(f"max relative R_nl deviation vs scipy: {worst:.3e} ({worst_ctx})")


@requires_scipy
def test_harmonic_orthonormality_through_the_python_api():
    """Orthonormality integrated through the public Python surface.

    `int conj(Y_l'm') Y_lm dOmega = delta`, quadratured from Python so that the
    check exercises the same values a user obtains. Simpson's rule is used along
    `theta` and the trapezoidal rule along `phi`; the latter is spectrally
    accurate for the periodic factor `exp(i (m2 - m1) phi)`, and with 400/160
    intervals the total quadrature error is far below the tolerance.

    An earlier version of this test used a coarse 200x200 trapezoid in both
    directions and failed at 1.0e-4 purely from quadrature error, with the
    implementation correct.
    """
    from scipy.integrate import simpson

    n_theta, n_phi = 400, 160
    thetas = np.linspace(0.0, np.pi, n_theta + 1)
    # Endpoint included: with an even number of intervals the trapezoidal rule
    # over a full period is exact for the periodic factor and for a constant.
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi + 1)
    states = [(l, m) for l in range(0, 3) for m in range(-l, l + 1)]

    table: dict[tuple[int, int], np.ndarray] = {}
    for l, m in states:
        vals = np.empty((thetas.size, phis.size), dtype=complex)
        for a, t in enumerate(thetas):
            for b, p in enumerate(phis):
                vals[a, b] = atoms.harmonics.value(l, m, float(t), float(p))
        table[(l, m)] = vals

    weights = np.sin(thetas)[:, None]
    for l1, m1 in states:
        for l2, m2 in states:
            integrand = (np.conj(table[(l1, m1)]) * table[(l2, m2)] * weights).real
            inner = np.trapezoid(integrand, phis, axis=1)
            integral = simpson(inner, x=thetas)
            expected = 1.0 if (l1, m1) == (l2, m2) else 0.0
            assert abs(integral - expected) < 1e-8, f"({l1},{m1})x({l2},{m2}): {integral}"


@requires_scipy
def test_addition_theorem_through_the_python_api():
    """`sum_m |Y_l^m(theta, phi)|^2 = (2l+1)/(4 pi)` exactly.

    A strong check of the relative phases between all `m` of a given `l`,
    evaluated through the public API.
    """
    for l in range(0, 7):
        expected = (2 * l + 1) / (4.0 * math.pi)
        for theta in (0.0, 0.31, 1.2, 2.4, math.pi):
            total = sum(
                atoms.harmonics.abs_squared(l, m, theta) for m in range(-l, l + 1)
            )
            assert abs(total - expected) < 1e-12 * max(expected, 1.0), (l, theta, total)
