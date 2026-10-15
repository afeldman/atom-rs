#!/usr/bin/env python3
"""Reproducible example: cross-sections of a hydrogen orbital.

Run it after `maturin develop`:

    python python/examples/orbital_explorer.py [output_directory]

It writes, into the output directory (default `python/examples/output/`):

* `density_2pz_xy.png`  — a cross-section through the 3D density
* `density_2pz_xz.png`
* `metadata.json`       — the reproducibility record for both images
* `summary.txt`         — the numbers printed below, as plain text

Only `numpy` and `atoms` are required: the PNG writer is implemented here with
the standard library so the example is hermetic and its output is byte-stable
across machines.

Everything is a pure function of the arguments and the seed in `SEED`, so two
runs produce identical files.
"""

from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

import atoms

# --- fixed parameters: nothing here is random except the sampling seed --------
N, L, M = 2, 1, 0  # 2p_z
EXTENT = (-14.0, 14.0)  # bohr
RESOLUTION = (141, 141, 141)
SEED = 20261015
N_SAMPLES = 20_000

# Five viridis anchor colours, linearly interpolated.
VIRIDIS = [
    (0.267004, 0.004874, 0.329415),
    (0.229739, 0.322361, 0.545706),
    (0.127568, 0.566949, 0.550556),
    (0.369214, 0.788888, 0.382914),
    (0.993248, 0.906157, 0.143936),
]


def colourise(normalised: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to RGB bytes."""
    v = np.clip(normalised, 0.0, 1.0)
    stops = np.array(VIRIDIS)
    pos = v * (len(stops) - 1)
    lo = np.floor(pos).astype(int).clip(0, len(stops) - 2)
    frac = (pos - lo)[..., None]
    rgb = stops[lo] * (1.0 - frac) + stops[lo + 1] * frac
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Write an 8-bit RGB PNG. `rgb` has shape (height, width, 3), uint8.

    Implemented with zlib and struct so that the example has no image-library
    dependency and produces byte-identical output for identical input.
    """
    height, width, _ = rgb.shape
    raw = b"".join(
        b"\x00" + rgb[y].tobytes() for y in range(height)  # filter byte 0 per row
    )

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- physics ------------------------------------------------------------
    hydrogen = atoms.Atom.hydrogen()
    orbital = hydrogen.orbital(n=N, l=L, m=M)
    grid = atoms.Grid3D(extent=EXTENT, resolution=RESOLUTION, unit="bohr")

    density = orbital.probability_density(grid)  # float64 (nx, ny, nz), bohr^-3
    total = orbital.total_probability(grid)

    print(f"package version       : {atoms.__version__}")
    print(f"model                 : hydrogenic-1e, unscreened, infinite nuclear mass")
    print(f"orbital               : {orbital.designation()}  {orbital.quantum_numbers()}")
    print(f"energy                : {orbital.energy('hartree'):.12f} hartree"
          f" = {orbital.energy('ev'):.9f} eV")
    print(f"analytic <r>          : {orbital.expectation_r():.6f} bohr")
    print(f"grid                  : {grid}")
    print(f"array shape / dtype   : {density.shape} / {density.dtype}"
          f" (C order, out[i, j, k] = (x_i, y_j, z_k))")
    print(f"cell volume           : {grid.volume_element_bohr3:.6e} bohr^3")
    print(f"max |psi|^2           : {density.max():.6e} bohr^-3")
    print(f"integrated |psi|^2    : {total:.6f}   (grid truncation + Riemann error)")

    # --- sampling ----------------------------------------------------------
    positions = atoms.sample_positions(orbital, N_SAMPLES, seed=SEED)
    radii = np.linalg.norm(positions, axis=1)
    print(f"sampled {N_SAMPLES} positions, seed {SEED}")
    print(f"  sampled <r>         : {radii.mean():.6f} bohr"
          f"   (analytic {orbital.expectation_r():.6f})")
    cos_sq = (positions[:, 2] / radii) ** 2
    print(f"  sampled <cos^2 th>  : {cos_sq.mean():.6f}   (analytic 0.6 for 2p_z)")

    # --- cross-sections -----------------------------------------------------
    x, y, z = grid.axes()
    k_mid = len(z) // 2           # z = 0 plane: identically zero for 2p_z
    j_mid = len(y) // 2           # y = 0 plane: the familiar dumbbell

    # Use a fixed dynamic range from the global maximum so images are comparable
    # across grids and across orbitals with the same normalisation.
    vmax = density.max()
    log_scale = np.log1p

    slices = {
        "density_2pz_xy.png": density[:, :, k_mid],      # z = 0 plane
        "density_2pz_xz.png": density[:, j_mid, :],      # y = 0 plane
    }
    for name, sl in slices.items():
        scaled = log_scale(sl / vmax * 20.0) / log_scale(21.0)
        rgb = colourise(scaled)
        # PNG rows run top-to-bottom; put positive y (or z) at the top.
        write_png(out_dir / name, np.flipud(rgb))
        print(f"wrote {out_dir / name}  ({sl.shape[1]}x{sl.shape[0]} px)")

    print(f"  z = 0 plane max     : {slices['density_2pz_xy.png'].max():.3e} bohr^-3"
          f"  (2p_z has a nodal plane here)")

    # --- reproducibility record --------------------------------------------
    meta = json.loads(
        atoms.metadata_json(
            "hydrogenic-1e (unscreened, infinite nuclear mass)",
            "bohr, hartree",
            seed=SEED,
            backend="cpu",
            notes=(
                "one electron, no screening, no electron correlation, "
                "no relativistic corrections"
            ),
            parameters=[
                ("orbital", f"n={N}, l={L}, m={M}"),
                ("grid_resolution", "x".join(str(n) for n in RESOLUTION)),
                ("grid_extent_bohr", f"{EXTENT[0]}..{EXTENT[1]}"),
                ("n_samples", str(N_SAMPLES)),
                ("integrated_probability", f"{total:.12f}"),
            ],
        )
    )
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    (out_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"orbital {orbital.designation()}",
                f"energy_hartree {orbital.energy('hartree'):.12f}",
                f"energy_ev {orbital.energy('ev'):.9f}",
                f"expectation_r_bohr {orbital.expectation_r():.9f}",
                f"integrated_probability {total:.12f}",
                f"sampled_mean_r_bohr {radii.mean():.9f}",
                f"seed {SEED}",
            ]
        )
        + "\n"
    )
    print(f"wrote {out_dir / 'metadata.json'}")
    print(f"wrote {out_dir / 'summary.txt'}")

    # The example is self-checking: these are documented properties, not
    # decoration. If the physics regresses, the example fails loudly.
    assert density.shape == tuple(RESOLUTION)
    assert density.dtype == np.float64
    assert density.min() >= 0.0
    assert total > 0.99, f"integrated probability {total} is too low"
    assert abs(radii.mean() - orbital.expectation_r()) < 0.15
    assert abs(cos_sq.mean() - 0.6) < 0.05, cos_sq.mean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
