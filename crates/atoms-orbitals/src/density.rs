//! Volumetric evaluation of probability densities.
//!
//! [`DensityGrid`] samples a [`HydrogenWavefunction`] on a
//! [`atoms_core::spatial_grid::SpatialGrid3D`] and offers the operations a
//! renderer or an analysis script needs: extrema, integrated probability,
//! isosurface thresholds, and 2D cross-sections.
//!
//! Evaluation is parallelised with `rayon` over grid points. Parallelism is
//! bit-for-bit deterministic here because each grid point is computed
//! independently and the reduction is a plain summation over a fixed order in
//! [`DensityGrid::total_probability`].

use crate::wavefunction::HydrogenWavefunction;
use atoms_core::spatial_grid::SpatialGrid3D;
use atoms_numerics::quadrature::simpson;
use rayon::prelude::*;

/// A sampled scalar field on a regular Cartesian grid.
#[derive(Debug, Clone, PartialEq)]
pub struct DensityGrid {
    grid: SpatialGrid3D,
    values: Vec<f64>,
}

/// Errors from constructing a [`DensityGrid`] from raw values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DensityGridError {
    /// The value buffer length did not match the grid point count.
    LengthMismatch {
        /// Number of grid points.
        expected: usize,
        /// Number of supplied values.
        got: usize,
    },
    /// A value was negative, which is impossible for a probability density.
    NegativeValue {
        /// Linear index of the offending value.
        index: usize,
    },
    /// A value was NaN or infinite.
    NonFinite {
        /// Linear index of the offending value.
        index: usize,
    },
}

impl std::fmt::Display for DensityGridError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DensityGridError::LengthMismatch { expected, got } => write!(
                f,
                "grid has {expected} points but {got} values were supplied"
            ),
            DensityGridError::NegativeValue { index } => {
                write!(f, "probability density at index {index} is negative")
            }
            DensityGridError::NonFinite { index } => {
                write!(f, "probability density at index {index} is not finite")
            }
        }
    }
}

impl std::error::Error for DensityGridError {}

impl DensityGrid {
    /// Sample `|psi|^2` at every grid point.
    pub fn probability_density(wf: &HydrogenWavefunction, grid: &SpatialGrid3D) -> Self {
        let n = grid.len();
        let values: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|lin| {
                let p = grid.point_at(lin);
                wf.probability_density(p.x, p.y, p.z)
            })
            .collect();
        Self {
            grid: grid.clone(),
            values,
        }
    }

    /// Wrap an existing value buffer, validating length and physical constraints.
    ///
    /// # Errors
    /// Returns [`DensityGridError`] if the buffer length does not match the
    /// grid, if any value is negative, or if any value is not finite.
    pub fn from_values(grid: SpatialGrid3D, values: Vec<f64>) -> Result<Self, DensityGridError> {
        if values.len() != grid.len() {
            return Err(DensityGridError::LengthMismatch {
                expected: grid.len(),
                got: values.len(),
            });
        }
        for (i, &v) in values.iter().enumerate() {
            if !v.is_finite() {
                return Err(DensityGridError::NonFinite { index: i });
            }
            if v < 0.0 {
                return Err(DensityGridError::NegativeValue { index: i });
            }
        }
        Ok(Self { grid, values })
    }

    /// The underlying grid.
    pub fn grid(&self) -> &SpatialGrid3D {
        &self.grid
    }

    /// The sampled values in the grid's linear order.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Number of sampled points.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Always `false`; a valid grid has at least eight points.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Largest sampled value.
    pub fn max(&self) -> f64 {
        self.values.iter().copied().fold(0.0_f64, f64::max)
    }

    /// Linear index of the largest sampled value.
    pub fn max_index(&self) -> usize {
        self.values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Value at grid point `(i, j, k)`.
    pub fn value_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.values[self.grid.index(i, j, k)]
    }

    /// Integrated probability `sum |psi|^2 dV`.
    ///
    /// This is the discrete approximation to `int |psi|^2 dV`. It approaches 1
    /// as the grid is refined and its extent is enlarged; it is deliberately
    /// *not* renormalised, so that the value can be used as a test of both the
    /// wavefunction and the grid.
    pub fn total_probability(&self) -> f64 {
        self.values.iter().sum::<f64>() * self.grid.cell_volume()
    }

    /// Value of the isosurface threshold placed at `fraction` of the maximum.
    pub fn threshold_fraction_of_max(&self, fraction: f64) -> f64 {
        self.max() * fraction
    }

    /// Slice at fixed `k` (constant `z`), returned as a buffer of `nx * ny`
    /// values with `i` varying fastest.
    ///
    /// # Panics
    /// Panics if `k >= nz`.
    pub fn cross_section_xy(&self, k: usize) -> Vec<f64> {
        let [nx, ny, nz] = self.grid.counts();
        assert!(k < nz, "k = {k} is outside [0, {nz})");
        let mut out = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                out.push(self.value_at(i, j, k));
            }
        }
        out
    }

    /// Slice at fixed `j` (constant `y`), returned as a buffer of `nx * nz`
    /// values with `i` varying fastest.
    ///
    /// # Panics
    /// Panics if `j >= ny`.
    pub fn cross_section_xz(&self, j: usize) -> Vec<f64> {
        let [nx, ny, nz] = self.grid.counts();
        assert!(j < ny, "j = {j} is outside [0, {ny})");
        let mut out = Vec::with_capacity(nx * nz);
        for k in 0..nz {
            for i in 0..nx {
                out.push(self.value_at(i, j, k));
            }
        }
        out
    }

    /// Radial coordinate of the maximum-probability sample, in bohr.
    ///
    /// This is a *sampled* quantity: it depends on the grid resolution and is
    /// not the analytic most-probable radius `n^2 + n(n-1)/2 ...`; use
    /// [`crate::expectations`] for analytic reference values.
    pub fn radius_of_max(&self) -> f64 {
        let p = self.grid.point_at(self.max_index());
        (p.x * p.x + p.y * p.y + p.z * p.z).sqrt()
    }
}

/// Numerical expectation value `<r^k>` in bohr^k.
///
/// Computed as `int_0^{r_max} r^k P(r) dr` with `P = r^2 R^2`, i.e. as
/// `int_0^{r_max} r^{k+2} R(r)^2 dr`, by composite Simpson quadrature on a
/// truncated domain. The truncation bound is chosen so that the neglected tail
/// is exponentially small, and the accuracy actually achieved is asserted in the
/// crate's tests against the closed forms in [`crate::expectations`].
///
/// `k` may be negative, which is used for `<1/r>` and `<1/r^2>`.
///
/// # Domain of validity
///
/// `k >= -2` is required. The integrand is `r^{k+2} R(r)^2`, so:
///
/// * `k = -2`: the integrand tends to `R(0)^2`, which is finite and non-zero for
///   s orbitals. This is handled exactly at `r = 0` rather than being replaced by
///   zero, which would bias the integral by `O(h)`.
/// * `k = -1`: the integrand vanishes at `r = 0`.
/// * `k < -2`: the integral diverges at the origin for s orbitals; `f64::NAN` is
///   returned.
pub fn radial_expectation_value(wf: &HydrogenWavefunction, k: i32) -> f64 {
    let power_of_r = k + 2;
    if power_of_r < 0 {
        // Divergent at the origin for s orbitals.
        return f64::NAN;
    }
    let n = wf.quantum_numbers().n() as f64;
    let r_max = 15.0 * n * n + 40.0;
    let intervals = 400_000;
    simpson(
        |r| {
            let rad = wf.radial_value(r);
            if power_of_r == 0 {
                // r^0 R^2, with 0^0 = 1 so that this is R(0)^2 at the origin.
                rad * rad
            } else {
                r.powi(power_of_r) * rad * rad
            }
        },
        0.0,
        r_max,
        intervals,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expectations as exact;
    use atoms_core::spatial_grid::SpatialGrid3D;
    use nalgebra::{Point3, Vector3};

    fn small_grid() -> SpatialGrid3D {
        SpatialGrid3D::new(
            Point3::new(-6.0, -6.0, -6.0),
            Vector3::new(0.25, 0.25, 0.25),
            [49, 49, 49],
        )
        .unwrap()
    }

    #[test]
    fn density_is_non_negative_and_finite_everywhere() {
        let grid = small_grid();
        for (n, l, m) in [(1u32, 0u32, 0i32), (2, 1, 1), (3, 2, -2), (4, 3, 3)] {
            let wf = HydrogenWavefunction::new(n, l, m).unwrap();
            let d = DensityGrid::probability_density(&wf, &grid);
            assert_eq!(d.len(), grid.len());
            for &v in d.values() {
                assert!(v.is_finite(), "n={n} l={l} m={m}");
                assert!(v >= 0.0);
            }
        }
    }

    #[test]
    fn values_are_laid_out_consistently_with_the_grid() {
        let grid = small_grid();
        let wf = HydrogenWavefunction::new(2, 1, 0).unwrap();
        let d = DensityGrid::probability_density(&wf, &grid);
        for k in 0..grid.counts()[2] {
            for j in 0..grid.counts()[1] {
                for i in 0..grid.counts()[0] {
                    let p = grid.point(i, j, k);
                    assert!(
                        (d.value_at(i, j, k) - wf.probability_density(p.x, p.y, p.z)).abs() < 1e-18
                    );
                }
            }
        }
    }

    /// Integrated probability on a large, fine grid must approach 1. The
    /// residual is a *grid* discretisation effect, not a normalisation error,
    /// and is reported explicitly.
    #[test]
    fn integrated_probability_approaches_one() {
        let grid = SpatialGrid3D::new(
            Point3::new(-20.0, -20.0, -20.0),
            Vector3::new(0.125, 0.125, 0.125),
            [321, 321, 321],
        )
        .unwrap();
        let wf = HydrogenWavefunction::new(1, 0, 0).unwrap();
        let d = DensityGrid::probability_density(&wf, &grid);
        let total = d.total_probability();
        assert!(
            (total - 1.0).abs() < 5e-3,
            "integrated probability = {total}"
        );
    }

    #[test]
    fn max_of_1s_density_is_at_the_origin() {
        let grid = SpatialGrid3D::new(
            Point3::new(-5.0, -5.0, -5.0),
            Vector3::new(0.1, 0.1, 0.1),
            [101, 101, 101],
        )
        .unwrap();
        let wf = HydrogenWavefunction::new(1, 0, 0).unwrap();
        let d = DensityGrid::probability_density(&wf, &grid);
        let p = grid.point_at(d.max_index());
        assert!(p.coords.norm() < 0.15, "argmax at {p}");
        // |psi(0)|^2 = 1/pi
        assert!((d.max() - 1.0 / std::f64::consts::PI).abs() < 1e-14);
    }

    #[test]
    fn cross_sections_have_the_documented_shape_and_layout() {
        let grid = small_grid();
        let wf = HydrogenWavefunction::new(3, 2, 0).unwrap();
        let d = DensityGrid::probability_density(&wf, &grid);
        let [nx, ny, nz] = grid.counts();
        let xy = d.cross_section_xy(10);
        assert_eq!(xy.len(), nx * ny);
        let xz = d.cross_section_xz(3);
        assert_eq!(xz.len(), nx * nz);
        // layout: element (i, j) of the xy slice is value_at(i, j, k)
        assert_eq!(xy[5 * nx + 7], d.value_at(7, 5, 10));
        assert_eq!(xz[5 * nx + 7], d.value_at(7, 3, 5));
    }

    #[test]
    fn from_values_validates_input() {
        let grid = small_grid();
        let ok = vec![0.0; grid.len()];
        assert!(DensityGrid::from_values(grid.clone(), ok).is_ok());

        let short = vec![0.0; grid.len() - 1];
        assert!(matches!(
            DensityGrid::from_values(grid.clone(), short),
            Err(DensityGridError::LengthMismatch { .. })
        ));

        let mut bad = vec![0.0; grid.len()];
        bad[3] = -1.0;
        assert!(matches!(
            DensityGrid::from_values(grid.clone(), bad),
            Err(DensityGridError::NegativeValue { index: 3 })
        ));

        let mut nan = vec![0.0; grid.len()];
        nan[7] = f64::NAN;
        assert!(matches!(
            DensityGrid::from_values(grid, nan),
            Err(DensityGridError::NonFinite { index: 7 })
        ));
    }

    #[test]
    fn inverse_r_moments_handle_the_origin_correctly() {
        // Regression test. The integrand for <r^k> is r^{k+2} R(r)^2. For
        // k = -2 it tends to R(0)^2, which is non-zero for s orbitals.
        // Substituting 0 for the r = 0 endpoint value biases the Simpson sum by
        // O(h) and produced <1/r^2> = 1.99982 instead of 2 for the 1s state.
        let wf = HydrogenWavefunction::new(1, 0, 0).unwrap();
        let inv_r2 = radial_expectation_value(&wf, -2);
        assert!(
            (inv_r2 - 2.0).abs() < 1e-9,
            "<1/r^2> for 1s = {inv_r2}, analytic 2.0"
        );
        // R(0)^2 = (2)^2 = 4, which is the endpoint value.
        assert!((wf.radial_value(0.0).powi(2) - 4.0).abs() < 1e-15);

        // k = -1: the integrand r R^2 vanishes at the origin.
        let inv_r = radial_expectation_value(&wf, -1);
        assert!((inv_r - 1.0).abs() < 1e-9, "<1/r> for 1s = {inv_r}");

        // k < -2 diverges at the origin for s orbitals.
        assert!(radial_expectation_value(&wf, -3).is_nan());
        assert!(radial_expectation_value(&wf, -5).is_nan());
    }

    #[test]
    fn inverse_square_moment_for_non_s_states() {
        // <1/r^2> = 1 / (n^3 (l + 1/2))
        for (n, l) in [(2u32, 1u32), (3, 1), (3, 2), (4, 3)] {
            let wf = HydrogenWavefunction::new(n, l, 0).unwrap();
            let got = radial_expectation_value(&wf, -2);
            let want = 1.0 / ((n as f64).powi(3) * (l as f64 + 0.5));
            assert!(
                (got - want).abs() < 1e-7 * want,
                "n={n} l={l}: {got} vs {want}"
            );
        }
    }

    /// The numerical `<r>` must match the analytic closed form; this is the
    /// quantitative link between [`radial_expectation_value`] and
    /// [`crate::expectations`].
    #[test]
    fn numerical_expectation_r_matches_closed_form() {
        for n in 1..=5u32 {
            for l in 0..n {
                let wf = HydrogenWavefunction::new(n, l, 0).unwrap();
                let numeric = radial_expectation_value(&wf, 1);
                let analytic = exact::expectation_r(n, l);
                assert!(
                    (numeric - analytic).abs() < 1e-6 * analytic.abs().max(1.0),
                    "n={n} l={l} numeric={numeric} analytic={analytic}"
                );
            }
        }
    }

    #[test]
    fn numerical_inverse_r_matches_closed_form() {
        for (n, l) in [(1u32, 0u32), (2, 0), (2, 1), (3, 2), (4, 0)] {
            let wf = HydrogenWavefunction::new(n, l, 0).unwrap();
            let numeric = radial_expectation_value(&wf, -1);
            let analytic = exact::expectation_inverse_r(n);
            assert!(
                (numeric - analytic).abs() < 1e-6 * analytic.abs(),
                "n={n} l={l} numeric={numeric} analytic={analytic}"
            );
        }
    }
}
