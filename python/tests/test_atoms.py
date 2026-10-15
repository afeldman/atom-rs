"""API contract tests for the `atoms` package.

These tests pin behaviour that the documentation promises: exception types,
array shapes, dtypes, memory order, axis orientation, units and determinism.
Numerical parity against the Rust core and SciPy lives in `test_parity.py`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import atoms


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_version_is_reported():
    assert isinstance(atoms.__version__, str)
    assert atoms.__version__.count(".") >= 1


def test_documented_names_are_present():
    for name in atoms.__all__:
        assert hasattr(atoms, name), name


def test_viewer_is_present_but_declares_it_is_unimplemented():
    # The signature is frozen now, the implementation is M2-C. It must fail
    # loudly rather than silently doing nothing.
    with pytest.raises(NotImplementedError) as exc:
        atoms.viewer.show(orbital=atoms.Atom.hydrogen().orbital(2, 1, 0),
                          quantity="probability_density")
    assert "M2-C" in str(exc.value)


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def test_unit_round_trips():
    for v in (0.0, 0.5, 1.0, 3.75, 100.0):
        assert atoms.units.angstrom_to_bohr(atoms.units.bohr_to_angstrom(v)) == pytest.approx(v)
        assert atoms.units.nanometer_to_bohr(atoms.units.bohr_to_nanometer(v)) == pytest.approx(v)
        assert atoms.units.ev_to_hartree(atoms.units.hartree_to_ev(v)) == pytest.approx(v)


def test_unit_reference_values():
    # 1 bohr = 0.529177210903 angstrom
    assert atoms.units.bohr_to_angstrom(1.0) == pytest.approx(0.529177210903, abs=1e-12)
    # 1 hartree = 27.211386245988 eV
    assert atoms.units.hartree_to_ev(1.0) == pytest.approx(27.211386245988, abs=1e-12)
    # 1 hartree = 2625.4996394799 kJ/mol
    assert atoms.units.hartree_to_kj_per_mol(1.0) == pytest.approx(2625.4996394799, abs=1e-9)
    # proton reduced-mass ratio
    assert atoms.units.reduced_mass_ratio(1836.15267344) == pytest.approx(0.9994556794247658,
                                                                         abs=1e-15)


def test_energy_level_helper_and_units():
    # 1s hydrogen: -0.5 hartree
    assert atoms.energy_level(1, 1, "hartree") == pytest.approx(-0.5, abs=1e-15)
    assert atoms.energy_level(1, 1, "ev") == pytest.approx(-13.605693122994, abs=1e-11)
    assert atoms.energy_level(2, 1, "hartree") == pytest.approx(-0.125, abs=1e-15)
    # hydrogenic scaling Z^2
    assert atoms.energy_level(1, 2, "hartree") == pytest.approx(-2.0, abs=1e-15)


def test_unknown_units_raise_value_error():
    with pytest.raises(ValueError):
        atoms.energy_level(1, 1, "furlong")
    with pytest.raises(ValueError):
        atoms.Grid3D(extent=(-1.0, 1.0), resolution=(4, 4, 4), unit="parsec")


# ---------------------------------------------------------------------------
# Atom
# ---------------------------------------------------------------------------


def test_hydrogen_basics():
    h = atoms.Atom.hydrogen()
    assert h.symbol == "H"
    assert h.z == 1
    assert h.mass_amu == pytest.approx(1.00727646688, abs=1e-9)
    assert h.hydrogenic_model_is_exact is True
    assert h.position == (0.0, 0.0, 0.0)
    assert "1s1" == h.nominal_configuration
    assert "Atom(" in repr(h)


def test_from_z_range_and_errors():
    for z in range(1, 11):
        assert atoms.Atom.from_z(z).z == z
    assert atoms.Atom.from_z(2).hydrogenic_model_is_exact is False
    for bad in (0, 11, 99):
        with pytest.raises(ValueError):
            atoms.Atom.from_z(bad)


def test_nominal_configuration_is_metadata_not_a_result():
    # documented as descriptive metadata; it must not be used by any calculation
    c = atoms.Atom.from_z(6)
    assert c.nominal_configuration == "[He] 2s2 2p2"
    assert c.hydrogenic_model_is_exact is False


# ---------------------------------------------------------------------------
# Quantum-number validation
# ---------------------------------------------------------------------------


def test_qn_validation_raises_value_error():
    h = atoms.Atom.hydrogen()
    with pytest.raises(ValueError):
        h.orbital(0, 0, 0)          # n must be >= 1
    with pytest.raises(ValueError):
        h.orbital(1, 1, 0)          # l < n required
    with pytest.raises(ValueError):
        h.orbital(2, 1, 2)          # |m| <= l required
    with pytest.raises(ValueError):
        h.orbital(2, 1, -2)


def test_negative_m_obeys_the_conjugate_phase_relation():
    """`psi_{n,l,-m} = (-1)^m conj(psi_{n,l,m})` and `|psi|` is `m`-symmetric.

    This is the actual convention, checked at general points. Note that at
    `phi = pi/2` the `m = +1` and `m = -1` values coincide, so a test that only
    samples that plane would not distinguish them.
    """
    h = atoms.Atom.hydrogen()
    for (n, l) in [(2, 1), (3, 2), (4, 3)]:
        for m in range(1, l + 1):
            for (x, y, z) in [(0.7, 0.3, 1.1), (2.0, 0.0, 0.0), (0.0, 1.3, -0.9)]:
                a = h.orbital(n, l, m).value_at(x, y, z)
                b = h.orbital(n, l, -m).value_at(x, y, z)
                expected = ((-1) ** m) * a.conjugate()
                assert abs(b - expected) < 1e-14, (n, l, m, x, y, z, a, b)
                assert abs(abs(a) - abs(b)) < 1e-14


def test_negative_and_positive_m_are_distinct():
    # At (2, 0, 0): r = 2, theta = pi/2, phi = 0.
    # Y_1^1 = -sqrt(3/8pi) and Y_1^{-1} = +sqrt(3/8pi): real, opposite signs.
    h = atoms.Atom.hydrogen()
    a = h.orbital(2, 1, 1).value_at(2.0, 0.0, 0.0)
    b = h.orbital(2, 1, -1).value_at(2.0, 0.0, 0.0)
    assert abs(a.imag) < 1e-12 and abs(b.imag) < 1e-12
    assert a.real * b.real < 0, (a, b)
    assert a != b
    # the densities, however, are identical
    assert h.orbital(2, 1, 1).probability_density_at(2.0, 0.0, 0.0) == pytest.approx(
        h.orbital(2, 1, -1).probability_density_at(2.0, 0.0, 0.0), rel=1e-15
    )


def test_qn_accessors_and_node_counts():
    o = atoms.Atom.hydrogen().orbital(4, 2, -1)
    assert (o.n, o.l, o.m, o.z) == (4, 2, -1, 1)
    assert o.quantum_numbers() == (4, 2, -1)
    assert o.radial_nodes == 1
    assert o.angular_nodes == 2
    assert o.total_nodes == 3
    assert o.designation() == "4d (m=-1)"
    assert atoms.Atom.hydrogen().orbital(2, 1, 0).designation() == "2p"


def test_hydrogenic_ion_from_species():
    he = atoms.Atom.from_z(2)
    o = he.orbital(1, 0, 0)
    assert o.z == 2
    assert o.energy("hartree") == pytest.approx(-2.0, abs=1e-15)


# ---------------------------------------------------------------------------
# Energies
# ---------------------------------------------------------------------------


def test_energies_match_the_closed_form():
    h = atoms.Atom.hydrogen()
    for n in range(1, 11):
        o = h.orbital(n, 0, 0)
        assert o.energy() == pytest.approx(-0.5 / n**2, abs=1e-15)
    assert h.orbital(1, 0, 0).energy("ev") == pytest.approx(-13.605693122994, abs=1e-11)
    assert h.orbital(1, 0, 0).energy("kj/mol") == pytest.approx(-1312.74981973995, abs=1e-8)


def test_energy_is_degenerate_in_l():
    h = atoms.Atom.hydrogen()
    for n in range(2, 6):
        reference = h.orbital(n, 0, 0).energy()
        for l in range(n):
            for m in range(-l, l + 1):
                assert h.orbital(n, l, m).energy() == pytest.approx(reference, abs=1e-15)


# ---------------------------------------------------------------------------
# Profiles: scalar vs array, dtypes
# ---------------------------------------------------------------------------


def test_scalar_and_array_profiles_agree():
    o = atoms.Atom.hydrogen().orbital(1, 0, 0)
    # 1s: R(r) = 2 e^{-r}
    assert o.radial_wavefunction(0.0) == pytest.approx(2.0, abs=1e-15)
    assert o.radial_wavefunction(1.0) == pytest.approx(2.0 * np.exp(-1.0), abs=1e-15)

    r = np.linspace(0.0, 8.0, 101)
    arr = o.radial_wavefunction(r)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == r.shape
    assert arr.dtype == np.float64
    assert arr == pytest.approx(2.0 * np.exp(-r), abs=1e-14)


def test_radial_probability_density_unit_and_origin():
    o = atoms.Atom.hydrogen().orbital(1, 0, 0)
    # P(0) = 0 for every orbital, including s orbitals
    assert o.radial_probability_density(0.0) == 0.0
    # 1s: P(r) = 4 r^2 e^{-2r}, maximum at r = 1 bohr
    assert o.radial_probability_density(1.0) == pytest.approx(4.0 * np.exp(-2.0), abs=1e-15)
    r = np.linspace(0.0, 10.0, 2001)
    p = o.radial_probability_density(r)
    # integrates to 1 over r
    assert np.trapezoid(p, r) == pytest.approx(1.0, abs=1e-6)
    assert p.min() >= 0.0


def test_angular_probability_density_normalisation():
    # int_0^pi |Y|^2 sin(theta) dtheta = 1/(2 pi)
    theta = np.linspace(0.0, np.pi, 20_001)
    for (l, m) in [(0, 0), (1, 0), (1, 1), (2, -2), (3, 3)]:
        o = atoms.Atom.hydrogen().orbital(l + 1, l, m)
        a = o.angular_probability_density(theta)
        assert a.dtype == np.float64
        integral = np.trapezoid(a * np.sin(theta), theta)
        assert integral == pytest.approx(1.0 / (2.0 * np.pi), abs=1e-9)


def test_profile_type_errors():
    o = atoms.Atom.hydrogen().orbital(2, 1, 0)
    with pytest.raises(TypeError):
        o.radial_wavefunction("not a number")
    with pytest.raises(TypeError):
        o.radial_probability_density([1.0, 2.0])     # list, not ndarray
    with pytest.raises(TypeError):
        o.angular_probability_density(np.zeros((2, 2)))  # wrong ndim


# ---------------------------------------------------------------------------
# Grid3D
# ---------------------------------------------------------------------------


def test_grid_geometry_cube_and_anisotropic():
    g = atoms.Grid3D(extent=(-8.0, 8.0), resolution=(11, 11, 11))
    assert g.unit == "bohr"
    assert g.resolution == (11, 11, 11)
    assert g.shape == (11, 11, 11)
    assert g.size == 11**3
    assert g.spacing == pytest.approx((1.6, 1.6, 1.6), abs=1e-15)
    assert g.volume_element_bohr3 == pytest.approx(1.6**3, abs=1e-12)
    e = np.array(g.extent)
    assert e[:, 0] == pytest.approx([-8, -8, -8])
    assert e[:, 1] == pytest.approx([8, 8, 8])

    a = atoms.Grid3D(extent=((-8.0, 12.0), (-6.0, 6.0), (-4.0, 4.0)), resolution=(11, 7, 5))
    assert a.resolution == (11, 7, 5)
    assert a.spacing == pytest.approx((2.0, 2.0, 2.0), abs=1e-15)


def test_grid_axes_match_the_spacing_convention():
    g = atoms.Grid3D(extent=((-8.0, 12.0), (-6.0, 6.0), (-4.0, 4.0)), resolution=(11, 7, 5))
    x, y, z = g.axes()
    assert len(x) == 11 and len(y) == 7 and len(z) == 5
    assert x[0] == pytest.approx(-8.0) and x[-1] == pytest.approx(12.0)
    assert y[0] == pytest.approx(-6.0) and y[-1] == pytest.approx(6.0)
    assert z[0] == pytest.approx(-4.0) and z[-1] == pytest.approx(4.0)
    assert x[4] == pytest.approx(0.0)
    assert y[3] == pytest.approx(0.0)
    assert z[2] == pytest.approx(0.0)


def test_grid_in_angstrom_is_converted_internally():
    # +/-2 bohr expressed in angstrom; physical outputs stay in bohr
    extent_angstrom = atoms.units.bohr_to_angstrom(2.0)
    g = atoms.Grid3D(extent=(-extent_angstrom, extent_angstrom), resolution=(3, 3, 3),
                     unit="angstrom")
    assert g.unit == "angstrom"
    e = g.bohr_extent
    assert e[0][0] == pytest.approx(-2.0, abs=1e-12)
    assert e[0][1] == pytest.approx(2.0, abs=1e-12)
    # spacing 4 bohr over 2 intervals = 2 bohr, reported back in angstrom
    assert g.spacing[0] == pytest.approx(extent_angstrom, abs=1e-12)
    assert g.bohr_spacing[0] == pytest.approx(2.0, abs=1e-12)
    # the reported extent in angstrom round-trips to what was requested
    assert g.extent[0][0] == pytest.approx(-extent_angstrom, abs=1e-12)
    assert g.extent[0][1] == pytest.approx(extent_angstrom, abs=1e-12)


def test_grid_invalid_arguments():
    with pytest.raises(ValueError):
        atoms.Grid3D(extent=(1.0, -1.0), resolution=(4, 4, 4))       # hi < lo
    with pytest.raises(ValueError):
        atoms.Grid3D(extent=(0.0, 0.0), resolution=(4, 4, 4))         # degenerate
    with pytest.raises(ValueError):
        atoms.Grid3D(extent=(-1.0, 1.0), resolution=(1, 4, 4))        # too few points
    with pytest.raises(ValueError):
        atoms.Grid3D(extent=((-1.0, 1.0), (-1.0, 1.0)), resolution=(4, 4, 4))
    with pytest.raises(ValueError):
        atoms.Grid3D(extent="cube", resolution=(4, 4, 4))
    with pytest.raises(ValueError):
        atoms.Grid3D(extent=(float("nan"), 1.0), resolution=(4, 4, 4))


# ---------------------------------------------------------------------------
# Density arrays: shape, dtype, order, orientation
# ---------------------------------------------------------------------------


def test_density_array_shape_dtype_and_order():
    g = atoms.Grid3D(extent=(-8.0, 8.0), resolution=(11, 9, 7))
    d = atoms.Atom.hydrogen().orbital(2, 1, 0).probability_density(g)
    assert isinstance(d, np.ndarray)
    assert d.shape == (11, 9, 7)
    assert d.dtype == np.float64
    assert d.flags["C_CONTIGUOUS"]
    assert np.all(d >= 0.0)


def test_density_array_index_matches_point_evaluation():
    # out[i, j, k] must be the value at (x_i, y_j, z_k)
    g = atoms.Grid3D(extent=((-8.0, 12.0), (-6.0, 6.0), (-4.0, 4.0)), resolution=(11, 7, 5))
    o = atoms.Atom.hydrogen().orbital(3, 2, -2)
    d = o.probability_density(g)
    x, y, z = g.axes()
    for i in (0, 4, 10):
        for j in (0, 3, 6):
            for k in (0, 2, 4):
                assert d[i, j, k] == pytest.approx(o.probability_density_at(x[i], y[j], z[k]),
                                                   rel=1e-13, abs=0.0)


def test_density_orientation_is_not_transposed():
    # 1s peaks at the origin. With an asymmetric, anisotropic grid the origin
    # sits at a specific (i, j, k); any transpose would move it.
    g = atoms.Grid3D(extent=((-8.0, 12.0), (-6.0, 6.0), (-4.0, 4.0)), resolution=(11, 7, 5))
    d = atoms.Atom.hydrogen().orbital(1, 0, 0).probability_density(g)
    idx = np.unravel_index(int(np.argmax(d)), d.shape)
    assert idx == (4, 3, 2), f"origin peak at {idx}, expected (4, 3, 2)"


def test_2p_z_vanishes_in_the_z_equals_zero_plane():
    # odd nz so that k = nz//2 is exactly z = 0
    g = atoms.Grid3D(extent=(-8.0, 8.0), resolution=(11, 11, 11))
    d = atoms.Atom.hydrogen().orbital(2, 1, 0).probability_density(g)
    assert np.all(d[:, :, 5] < 1e-25)
    assert np.max(d) > 1e-3


def test_2p_m_one_vanishes_on_the_z_axis_but_not_in_the_mid_plane():
    g = atoms.Grid3D(extent=(-8.0, 8.0), resolution=(11, 11, 11))
    d = atoms.Atom.hydrogen().orbital(2, 1, 1).probability_density(g)
    # z axis is (i, j) = (5, 5)
    assert np.all(d[5, 5, :] < 1e-25)
    # but the z = 0 plane does contain density
    assert np.max(d[:, :, 5]) > 1e-6


def test_total_probability_and_volume_element():
    g = atoms.Grid3D(extent=(-12.0, 12.0), resolution=(97, 97, 97))
    o = atoms.Atom.hydrogen().orbital(1, 0, 0)
    d = o.probability_density(g)
    assert float(d.sum()) * g.volume_element_bohr3 == pytest.approx(o.total_probability(g),
                                                                   rel=1e-14)
    assert o.total_probability(g) > 0.99


def test_density_of_m_and_minus_m_have_equal_magnitude():
    g = atoms.Grid3D(extent=(-10.0, 10.0), resolution=(21, 21, 21))
    h = atoms.Atom.hydrogen()
    for (n, l) in [(2, 1), (3, 2), (4, 3)]:
        for m in range(1, l + 1):
            a = h.orbital(n, l, m).probability_density(g)
            b = h.orbital(n, l, -m).probability_density(g)
            assert np.allclose(a, b, rtol=1e-14, atol=0.0)


def test_density_on_a_grid_wrong_type_raises():
    o = atoms.Atom.hydrogen().orbital(1, 0, 0)
    with pytest.raises(TypeError):
        o.probability_density((-4.0, 4.0))


# ---------------------------------------------------------------------------
# Expectation values
# ---------------------------------------------------------------------------


def test_analytic_expectations_match_closed_forms():
    h = atoms.Atom.hydrogen()
    o = h.orbital(1, 0, 0)
    assert o.expectation_r() == pytest.approx(1.5, abs=1e-15)
    assert o.expectation_r_squared() == pytest.approx(3.0, abs=1e-15)
    assert o.expectation_inverse_r() == pytest.approx(1.0, abs=1e-15)
    assert o.expectation_inverse_r_squared() == pytest.approx(2.0, abs=1e-15)

    p = h.orbital(2, 1, 0)
    assert p.expectation_r() == pytest.approx(5.0, abs=1e-15)
    assert p.expectation_r_squared() == pytest.approx(30.0, abs=1e-15)
    assert p.expectation_inverse_r_squared() == pytest.approx(1.0 / 12.0, abs=1e-15)


def test_numerical_expectation_matches_analytic():
    o = atoms.Atom.hydrogen().orbital(3, 1, 0)
    assert o.radial_expectation_value(1) == pytest.approx(o.expectation_r(), rel=1e-6)
    assert o.radial_expectation_value(-1) == pytest.approx(o.expectation_inverse_r(), rel=1e-6)


def test_divergent_expectation_raises():
    o = atoms.Atom.hydrogen().orbital(1, 0, 0)
    with pytest.raises(ValueError):
        o.radial_expectation_value(-3)


# ---------------------------------------------------------------------------
# Error messages are specific
# ---------------------------------------------------------------------------


def test_error_messages_name_the_violation():
    h = atoms.Atom.hydrogen()
    with pytest.raises(ValueError, match="m = 5"):
        h.orbital(2, 1, 5)
    with pytest.raises(ValueError, match=r"l < n|l = 1"):
        h.orbital(1, 1, 0)
    with pytest.raises(ValueError, match="at least 2"):
        atoms.Grid3D(extent=(-1.0, 1.0), resolution=(1, 4, 4))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sample_radial_shape_dtype_and_determinism():
    o = atoms.Atom.hydrogen().orbital(1, 0, 0)
    r = atoms.sample_radial(o, 5000, seed=20261015)
    assert isinstance(r, np.ndarray)
    assert r.shape == (5000,)
    assert r.dtype == np.float64
    assert np.all(r >= 0.0)
    again = atoms.sample_radial(o, 5000, seed=20261015)
    assert np.array_equal(r, again)
    other = atoms.sample_radial(o, 5000, seed=20261016)
    assert not np.array_equal(r, other)


def test_sample_radial_matches_analytic_moments():
    o = atoms.Atom.hydrogen().orbital(1, 0, 0)
    r = atoms.sample_radial(o, 200_000, seed=1)
    assert r.mean() == pytest.approx(1.5, abs=0.02)
    assert r.var() == pytest.approx(0.75, abs=0.03)


def test_sample_positions_shape_and_radial_scale():
    o = atoms.Atom.hydrogen().orbital(2, 1, 0)
    p = atoms.sample_positions(o, 20_000, seed=7)
    assert p.shape == (20_000, 3)
    assert p.dtype == np.float64
    assert p.flags["C_CONTIGUOUS"]
    r = np.linalg.norm(p, axis=1)
    assert r.mean() == pytest.approx(o.expectation_r(), abs=0.1)


def test_sample_invalid_arguments():
    o = atoms.Atom.hydrogen().orbital(1, 0, 0)
    with pytest.raises(ValueError):
        atoms.sample_radial(o, 0)
    with pytest.raises(ValueError):
        atoms.sample_positions(o, 0)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_json_round_trip_and_reproducibility_fields():
    s = atoms.metadata_json(
        "hydrogenic-1e (unscreened, infinite nuclear mass)",
        "bohr, hartree",
        seed=20261015,
        backend="cpu",
        notes="no screening, no electron correlation",
        parameters=[("grid", "97x97x97")],
    )
    d = json.loads(s)
    assert d["units"] == "bohr, hartree"
    assert d["random_seed"] == 20261015
    assert d["backend"] == "cpu"
    assert d["software_version"] == atoms.__version__
    assert d["numerical_parameters"] == [["grid", "97x97x97"]]
    # absent fields are null, never fabricated
    assert d["basis_set"] is None
    assert d["convergence"] is None


def test_metadata_defaults_are_null():
    d = json.loads(atoms.metadata_json("model", "au"))
    assert d["basis_set"] is None
    assert d["random_seed"] is None
    assert d["backend"] is None
    assert d["notes"] is None
