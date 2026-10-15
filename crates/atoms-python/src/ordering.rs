//! Array-ordering helpers shared by the binding layer.
//!
//! This module contains no PyO3 types, so it is compiled and tested by plain
//! `cargo test -p atoms-python` with the `python` feature disabled.
//!
//! # Why a separate ordering step exists
//!
//! The Rust grid stores values in C order over `(i, j, k)` with **`i` fastest**:
//!
//! ```text
//! rust_index(i, j, k) = (k * ny + j) * nx + i
//! ```
//!
//! NumPy users expect an array of shape `(nx, ny, nz)` in C order, i.e. **`k`
//! fastest**:
//!
//! ```text
//! numpy_index(i, j, k) = (i * ny + j) * nz + k
//! ```
//!
//! Handing the Rust buffer to NumPy unchanged would silently present the array
//! transposed. Rather than transpose after the fact (which needs either a copy or
//! a non-contiguous view), the evaluation is performed directly in the NumPy
//! order. One allocation, no transpose, no extra pass.
//!
//! `atoms_validation`'s integration tests pin the Rust layout; the tests in this
//! module pin the NumPy layout and their mutual consistency.

use atoms_core::spatial_grid::SpatialGrid3D;
use atoms_orbitals::HydrogenWavefunction;
use rayon::prelude::*;

/// Evaluate `|psi|^2` on `grid`, returning values in NumPy C order for an array
/// of shape `(nx, ny, nz)`.
///
/// The result satisfies
/// `out[(i * ny + j) * nz + k] == wf.probability_density(x_i, y_j, z_k)`
/// in bohr, i.e. `out` can be handed to `numpy` with `shape = (nx, ny, nz)`
/// without reshaping or transposing.
///
/// Evaluation is parallelised over the `x` index; each output position is
/// written by exactly one task through its own row buffer, so the result is
/// bit-for-bit deterministic.
pub fn probability_density_numpy_order(
    wf: &HydrogenWavefunction,
    grid: &SpatialGrid3D,
) -> Vec<f64> {
    let [nx, ny, nz] = grid.counts();

    // One contiguous row per (i, j) pair, in the order NumPy expects.
    let rows: Vec<Vec<f64>> = (0..nx)
        .into_par_iter()
        .map(|i| {
            let mut block = Vec::with_capacity(ny * nz);
            for j in 0..ny {
                for k in 0..nz {
                    let p = grid.point(i, j, k);
                    block.push(wf.probability_density(p.x, p.y, p.z));
                }
            }
            block
        })
        .collect();

    let mut out = Vec::with_capacity(nx * ny * nz);
    for block in rows {
        out.extend_from_slice(&block);
    }
    debug_assert_eq!(out.len(), nx * ny * nz);
    out
}

/// Index into a buffer produced by [`probability_density_numpy_order`].
///
/// Delegates to `atoms_core::spatial_grid::public_linear_index`, the workspace's
/// single definition of the public array order.
pub fn numpy_linear_index(i: usize, j: usize, k: usize, counts: [usize; 3]) -> usize {
    atoms_core::spatial_grid::public_linear_index(i, j, k, counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use atoms_orbitals::DensityGrid;
    use nalgebra::{Point3, Vector3};

    fn grid() -> SpatialGrid3D {
        // Deliberately anisotropic: an accidental transpose cannot pass.
        SpatialGrid3D::new(
            Point3::new(-4.0, -8.0, -2.0),
            Vector3::new(0.5, 0.25, 0.125),
            [17, 33, 9],
        )
        .unwrap()
    }

    #[test]
    fn numpy_order_matches_the_documented_index_map() {
        let g = grid();
        let wf = HydrogenWavefunction::new(2, 1, 0).unwrap();
        let vals = probability_density_numpy_order(&wf, &g);
        let counts = g.counts();
        assert_eq!(vals.len(), counts[0] * counts[1] * counts[2]);

        for i in 0..counts[0] {
            for j in 0..counts[1] {
                for k in 0..counts[2] {
                    let p = g.point(i, j, k);
                    let expected = wf.probability_density(p.x, p.y, p.z);
                    let idx = numpy_linear_index(i, j, k, counts);
                    assert_eq!(vals[idx], expected, "i={i} j={j} k={k}");
                }
            }
        }
    }

    /// The orientation must be observable, not merely self-consistent: the
    /// arrangement must be the one the physics predicts, in the axes the
    /// documentation names.
    ///
    /// `2p_z` has `|psi|^2 prop z^2 exp(-|z|)` on the `z` axis (with
    /// `R_21(r)^2 = r^2 e^{-r}/24` and `|Y_1^0|^2 = 3/(4 pi)` there), so its
    /// maximum lies at `z = +-2` bohr — an *interior* `k`, not an extreme one.
    /// The maximum must also sit on the `z` axis, i.e. at the central `i` and
    /// `j`.
    #[test]
    fn orientation_matches_the_physics() {
        let g = SpatialGrid3D::new(
            Point3::new(-5.0, -5.0, -5.0),
            Vector3::new(0.5, 0.5, 0.5),
            [21, 21, 21],
        )
        .unwrap();
        let wf = HydrogenWavefunction::new(2, 1, 0).unwrap();
        let vals = probability_density_numpy_order(&wf, &g);
        let counts = g.counts();

        let mut best = (0usize, 0usize, 0usize, 0.0f64);
        for i in 0..counts[0] {
            for j in 0..counts[1] {
                for k in 0..counts[2] {
                    let v = vals[numpy_linear_index(i, j, k, counts)];
                    if v > best.3 {
                        best = (i, j, k, v);
                    }
                }
            }
        }
        assert_eq!(
            best.0,
            counts[0] / 2,
            "i should be the centre, got {best:?}"
        );
        assert_eq!(
            best.1,
            counts[1] / 2,
            "j should be the centre, got {best:?}"
        );
        let z_at_max = g.point(best.0, best.1, best.2).z;
        assert!(
            (z_at_max.abs() - 2.0).abs() <= 0.5,
            "the 2p_z maximum should be near |z| = 2 bohr, got z = {z_at_max} ({best:?})"
        );

        // and the z = 0 plane vanishes identically
        let k_mid = counts[2] / 2;
        assert!(g.point(0, 0, k_mid).z.abs() < 1e-12, "k_mid must be z = 0");
        for i in 0..counts[0] {
            for j in 0..counts[1] {
                assert!(vals[numpy_linear_index(i, j, k_mid, counts)] < 1e-25);
            }
        }
    }

    /// Reordering must not change the physics, and it must not change the
    /// *values* at all.
    ///
    /// Two separate statements are checked:
    ///
    /// * every value is bit-for-bit identical once mapped to the same grid point
    ///   (the two buffers differ only in memory order);
    /// * the integrated probability agrees to `1e-12`. The two sums accumulate
    ///   in different orders — one in NumPy order, one in Rust grid order — so
    ///   they differ by floating-point summation error of order
    ///   `N * eps * <term>`, which for `81^3` terms is around `1e-13`. The
    ///   tolerance is stated rather than set to zero.
    #[test]
    fn reordering_changes_layout_only_not_values() {
        let g = SpatialGrid3D::new(
            Point3::new(-10.0, -10.0, -10.0),
            Vector3::new(0.25, 0.25, 0.25),
            [81, 81, 81],
        )
        .unwrap();
        let counts = g.counts();
        for (n, l, m) in [(1u32, 0u32, 0i32), (2, 1, 1), (3, 2, -2)] {
            let wf = HydrogenWavefunction::new(n, l, m).unwrap();
            let a = probability_density_numpy_order(&wf, &g);
            let b = DensityGrid::probability_density(&wf, &g);

            // identical values, different layout
            let mut max_dev = 0.0_f64;
            for i in 0..counts[0] {
                for j in 0..counts[1] {
                    for k in 0..counts[2] {
                        let va = a[numpy_linear_index(i, j, k, counts)];
                        let vb = b.values()[g.index(i, j, k)];
                        max_dev = max_dev.max((va - vb).abs());
                    }
                }
            }
            assert_eq!(max_dev, 0.0, "n={n} l={l} m={m}: values differ");

            let total_a: f64 = a.iter().sum::<f64>() * g.cell_volume();
            let total_b = b.total_probability();
            assert!(
                (total_a - total_b).abs() < 1e-12,
                "n={n} l={l} m={m}: {total_a} vs {total_b}"
            );
        }
    }

    #[test]
    fn parallel_evaluation_is_deterministic() {
        let g = grid();
        let wf = HydrogenWavefunction::new(3, 2, -1).unwrap();
        let a = probability_density_numpy_order(&wf, &g);
        let b = probability_density_numpy_order(&wf, &g);
        assert_eq!(a, b);
    }

    #[test]
    fn numpy_index_map_is_unique_and_dense() {
        let g = grid();
        let counts = g.counts();
        let mut seen = vec![false; counts[0] * counts[1] * counts[2]];
        for i in 0..counts[0] {
            for j in 0..counts[1] {
                for k in 0..counts[2] {
                    let idx = numpy_linear_index(i, j, k, counts);
                    assert!(!seen[idx], "duplicate index {idx}");
                    seen[idx] = true;
                }
            }
        }
        assert!(seen.iter().all(|&b| b));
    }
}
