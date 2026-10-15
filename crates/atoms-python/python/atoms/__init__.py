"""``atoms`` — hydrogen-like quantum orbitals, densities and sampling.

The numerics live in Rust (see the ``atom-rs`` repository); this package is a
thin, documented Python layer over the compiled extension module
:mod:`atoms._atoms`.

Conventions
-----------
* **Units.** All physical quantities are returned in Hartree atomic units:
  lengths in bohr, energies in hartree, probability density in ``bohr^-3``,
  radial probability density in ``bohr^-1``, angular probability density in
  ``sr^-1``. Pass ``unit=`` where an alternative is accepted, and use
  :mod:`atoms.units` to convert results.
* **Angles.** ``theta`` is the polar angle measured from ``+z`` in ``[0, pi]``;
  ``phi`` is the azimuth measured from ``+x`` towards ``+y`` in ``[0, 2 pi)``.
  This is the physics convention, matching ``scipy.special.sph_harm_y``.
* **Arrays.** ``Orbital.probability_density(grid)`` returns a ``float64`` array
  of shape ``(nx, ny, nz)`` in C order, where ``out[i, j, k]`` is the value at
  ``(x_i, y_j, z_k)``. Multiply by ``grid.volume_element_bohr3`` for probability
  per cell. Sampling returns ``float64`` arrays: ``(n,)`` radii or ``(n, 3)``
  Cartesian positions.
* **Complex wavefunctions.** ``m != 0`` orbitals are genuinely complex;
  ``Orbital.value_at`` returns a Python ``complex`` and
  ``Orbital.probability_density*`` returns the real ``|psi|^2``.

Scope
-----
Only one-electron, unscreened hydrogenic states are implemented. ``Z > 1`` gives
the unscreened hydrogenic approximation. No molecular electronic-structure
method, basis set, or molecular property is provided.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ._atoms import (
    Atom,
    Grid3D,
    Orbital,
    __version__,
    angstrom_to_bohr,
    bohr_to_angstrom,
    bohr_to_nanometer,
    energy_level,
    ev_to_hartree,
    hartree_to_ev,
    hartree_to_joule,
    hartree_to_kj_per_mol,
    metadata_json,
    nanometer_to_bohr,
    reduced_mass_ratio,
    sample_positions,
    sample_radial,
    spherical_harmonic,
    spherical_harmonic_abs_squared,
)

__all__ = [
    "Atom",
    "Orbital",
    "Grid3D",
    "units",
    "sampling",
    "sample_radial",
    "sample_positions",
    "metadata_json",
    "energy_level",
    "reduced_mass_ratio",
    "harmonics",
    "viewer",
    "__version__",
]

#: Unit conversions. Each function takes and returns a float; vectorise with
#: NumPy where needed.
#:
#: ``bohr_to_angstrom``, ``angstrom_to_bohr``, ``bohr_to_nanometer``,
#: ``nanometer_to_bohr``, ``hartree_to_ev``, ``ev_to_hartree``,
#: ``hartree_to_kj_per_mol``, ``hartree_to_joule``, ``reduced_mass_ratio``.
units = SimpleNamespace(
    bohr_to_angstrom=bohr_to_angstrom,
    angstrom_to_bohr=angstrom_to_bohr,
    bohr_to_nanometer=bohr_to_nanometer,
    nanometer_to_bohr=nanometer_to_bohr,
    hartree_to_ev=hartree_to_ev,
    ev_to_hartree=ev_to_hartree,
    hartree_to_kj_per_mol=hartree_to_kj_per_mol,
    hartree_to_joule=hartree_to_joule,
    reduced_mass_ratio=reduced_mass_ratio,
)


#: Spherical harmonics. `spherical_harmonic(l, m, theta, phi)` returns a Python
#: `complex`; `spherical_harmonic_abs_squared(l, m, theta)` returns `|Y|^2` in
#: `sr^-1`. Angles in radians, `theta` polar from `+z`.
harmonics = SimpleNamespace(
    value=spherical_harmonic,
    abs_squared=spherical_harmonic_abs_squared,
)

#: Reproducible sampling. `radial(orbital, n_samples, seed)` returns radii in
#: bohr; `positions(orbital, n_samples, seed)` returns `(n, 3)` Cartesian
#: positions in bohr. Both are pure functions of the seed.
sampling = SimpleNamespace(
    radial=sample_radial,
    positions=sample_positions,
)


def _viewer_show(
    orbital: Orbital | None = None,
    quantity: str = "probability_density",
    **kwargs: Any,
) -> None:
    raise NotImplementedError(
        "atoms.viewer.show is not implemented yet. The interactive viewer is "
        "scheduled for milestone M2-C; until then use "
        "Orbital.probability_density(grid) and plot the array directly."
    )


#: The interactive viewer entry point.
#:
#: `show(orbital=..., quantity=...)` is the frozen signature, but the function
#: currently raises `NotImplementedError`: no viewer exists yet. It is present
#: so that the intended API is visible and callers fail loudly rather than
#: silently doing nothing.
viewer = SimpleNamespace(show=_viewer_show)
