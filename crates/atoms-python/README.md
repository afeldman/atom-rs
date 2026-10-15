# atoms

Python bindings for the `atom-rs` hydrogen-like quantum orbital library.

```python
import atoms

hydrogen = atoms.Atom.hydrogen()
orbital = hydrogen.orbital(n=2, l=1, m=0)

grid = atoms.Grid3D(extent=(-12.0, 12.0), resolution=(96, 96, 96), unit="bohr")
density = orbital.probability_density(grid)          # float64 (96, 96, 96)

print(orbital.energy(unit="ev"), "eV")               # -3.4 eV
print(density.sum() * grid.volume_element_bohr3)     # ~1
```

All quantities are in Hartree atomic units. See the package docstring and
`docs/python-api.md` in the repository for the full contract, including axis
order and dtypes.

## Scope

One-electron, unscreened hydrogenic states only. No molecular electronic
structure, no Hartree-Fock, no DFT, no basis sets.

Build from the repository:

```bash
python -m venv .venv && .venv/bin/pip install maturin pytest numpy
cd crates/atoms-python && ../../.venv/bin/maturin develop --release
```
