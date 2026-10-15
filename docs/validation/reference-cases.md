# Scientific Validation Reference Cases

This directory contains analytical solutions, numerical reference values, and test specifications for validating quantum mechanical calculations.

## 1. Hydrogen Atom Energy Eigenvalues

**Formula:** E_n = -0.5 / n² Hartree = -27.211396 eV / n²

| n | Hartree | eV     |
|---|---------|--------|
| 1 | -0.5    | -27.2114 |
| 2 | -0.125  | -6.8029  |
| 3 | -0.0556 | -3.0668  |
| 4 | -0.0313 | -1.7069  |

## 2. Radial Wavefunction Normalization

The radial wavefunction R_{n,l}(r) must satisfy:
∫₀^∞ r² |R_{n,l}(r)|² dr = 1

**Test:** For n=1, l=0 (1s orbital), compute numerical integral and verify normalization within floating-point tolerance.

## 3. Spherical Harmonics Orthonormality

Complex spherical harmonics Y_l^m(θ, φ) must satisfy:
∫₀^π ∫₀^{2π} Y*_{l',m'}(θ,φ) Y_{lm}(θ,φ) sinθ dφ dθ = δ_{ll'} δ_{mm'}

**Test:** Verify orthonormality for all combinations of (l,m) with 0 ≤ l ≤ 3, -l ≤ m ≤ l.

## 4. Probability Density Properties

For any orbital ψ(r,θ,φ):
- |ψ|² ≥ 0 everywhere
- ∫ |ψ|² dV = 1 (normalization)

Radial probability density: P(r) = r² |R_{n,l}(r)|²
- Must be ≥ 0
- Integrates to 1 over all r

## 5. Nodal Structure Verification

**1s orbital (n=1, l=0):** No angular nodes, no radial nodes

**2p orbitals (n=2, l=1):**
- m=0: nodal plane at θ = π/2 (xy-plane)
- m=±1: nodal plane at φ = 0 and φ = π (xz-plane)

## 6. Reference Implementation Comparison

For more complex systems (H₂, multi-electron atoms), compare against:

- NIST Computational Chemistry Comparison Database
- Gaussian/Psi4 benchmark results
- Established textbook examples (Griffiths, Shankar, Cohen-Tannoudji)

---

## Test Cases Directory Structure

```text
docs/validation/
├── hydrogen_energy_eigenvalues.md
├── radial_wavefunction_normalization.md
├── spherical_harmonics_orthonormality.md
└── nodal_structure_tests.md
```
