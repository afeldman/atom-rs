//! Regular Cartesian grids for volumetric evaluation.
//!
//! A grid is defined by an origin, a per-axis spacing, and a per-axis count of
//! points. All lengths are in bohr. Grids are *not* required to be cubic; the
//! three axes are independent.
//!
//! Grids only describe where to evaluate a quantity. They contain no physics
//! and no rendering state, so they are usable from the pure-Rust core without
//! any graphics dependency.

use nalgebra::{Point3, Vector3};
use serde::{Deserialize, Serialize};

/// Linear index into a buffer laid out in the **public array order**.
///
/// This free function is the single definition of the public layout for the whole
/// workspace. The order is C order over `(i, j, k)` with **`k` fastest**:
///
/// ```text
/// public_linear_index(i, j, k, [nx, ny, nz]) = (i * ny + j) * nz + k
/// ```
///
/// [`SpatialGrid3D::index_public`] calls it, and so does the Python binding's
/// ordering module — so the NumPy arrays the Python API returns and the volumes
/// the renderer consumes cannot drift apart. `atoms-render` cannot call it (it
/// depends on no physics crate by design) and implements the same formula
/// independently; the renderer's own tests pin that layout with an asymmetric
/// grid whose values encode their coordinates, and the orbital integration test
/// confirms the two agree on real data.
///
/// This is the **reverse** of [`SpatialGrid3D::index`], the internal layout with
/// `i` fastest.
pub const fn public_linear_index(i: usize, j: usize, k: usize, counts: [usize; 3]) -> usize {
    let [_nx, ny, nz] = counts;
    (i * ny + j) * nz + k
}

/// A regular Cartesian grid of sample points.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialGrid3D {
    origin: Point3<f64>,
    spacing: Vector3<f64>,
    counts: [usize; 3],
}

/// Errors that can occur when constructing a grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GridError {
    /// At least one axis had fewer than two points.
    TooFewPoints {
        /// The offending axis (0 = x, 1 = y, 2 = z).
        axis: usize,
        /// The requested count.
        count: usize,
    },
    /// A spacing was zero or non-finite.
    InvalidSpacing {
        /// The offending axis.
        axis: usize,
    },
    /// The origin was not finite.
    InvalidOrigin,
}

impl std::fmt::Display for GridError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GridError::TooFewPoints { axis, count } => write!(
                f,
                "axis {axis} has {count} points; a grid needs at least 2 points per axis"
            ),
            GridError::InvalidSpacing { axis } => {
                write!(f, "axis {axis} has a zero or non-finite spacing")
            }
            GridError::InvalidOrigin => write!(f, "grid origin is not finite"),
        }
    }
}

impl std::error::Error for GridError {}

impl SpatialGrid3D {
    /// Build a grid from an origin, per-axis spacing (bohr) and per-axis counts.
    ///
    /// # Errors
    /// Returns [`GridError`] if any axis has fewer than two points, if any
    /// spacing is zero or non-finite, or if the origin is not finite.
    pub fn new(
        origin: Point3<f64>,
        spacing: Vector3<f64>,
        counts: [usize; 3],
    ) -> Result<Self, GridError> {
        if !origin.x.is_finite() || !origin.y.is_finite() || !origin.z.is_finite() {
            return Err(GridError::InvalidOrigin);
        }
        for axis in 0..3 {
            if counts[axis] < 2 {
                return Err(GridError::TooFewPoints {
                    axis,
                    count: counts[axis],
                });
            }
            let s = spacing[axis];
            if !s.is_finite() || s == 0.0 {
                return Err(GridError::InvalidSpacing { axis });
            }
        }
        Ok(Self {
            origin,
            spacing,
            counts,
        })
    }

    /// Build a grid of `counts` points per axis spanning `[-half_extent, +half_extent]`
    /// on each axis, centred on `origin`. Spacing is chosen so that the outermost
    /// samples fall exactly on the extent.
    ///
    /// `half_extent` is in bohr.
    pub fn centred(
        origin: Point3<f64>,
        half_extent: [f64; 3],
        counts: [usize; 3],
    ) -> Result<Self, GridError> {
        let mut spacing = Vector3::zeros();
        for axis in 0..3 {
            let n = counts[axis].max(2);
            spacing[axis] = 2.0 * half_extent[axis] / (n as f64 - 1.0);
        }
        let start = Point3::new(
            origin.x - half_extent[0],
            origin.y - half_extent[1],
            origin.z - half_extent[2],
        );
        Self::new(start, spacing, counts)
    }

    /// Axis-aligned origin (first sample), in bohr.
    pub fn origin(&self) -> Point3<f64> {
        self.origin
    }

    /// Per-axis spacing in bohr.
    pub fn spacing(&self) -> Vector3<f64> {
        self.spacing
    }

    /// Number of points per axis.
    pub fn counts(&self) -> [usize; 3] {
        self.counts
    }

    /// Total number of sample points.
    pub fn len(&self) -> usize {
        self.counts[0] * self.counts[1] * self.counts[2]
    }

    /// Always `false`: a valid grid has at least `2 * 2 * 2` points.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Linear index of grid point `(i, j, k)`, with `i` varying fastest.
    ///
    /// The layout is C-order over `(i, j, k)`, which matches NumPy's default
    /// `order="C"` for an array shaped `(nx, ny, nz)` **only** when the array is
    /// indexed as `[k, j, i]`. This is documented explicitly because the
    /// Python bridge must not silently transpose data.
    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        debug_assert!(i < self.counts[0] && j < self.counts[1] && k < self.counts[2]);
        (k * self.counts[1] + j) * self.counts[0] + i
    }

    /// The `(i, j, k)` multi-index of a linear index.
    pub fn multi_index(&self, linear: usize) -> (usize, usize, usize) {
        let i = linear % self.counts[0];
        let rest = linear / self.counts[0];
        let j = rest % self.counts[1];
        let k = rest / self.counts[1];
        (i, j, k)
    }

    /// Cartesian coordinate of grid point `(i, j, k)`, in bohr.
    pub fn point(&self, i: usize, j: usize, k: usize) -> Point3<f64> {
        Point3::new(
            self.origin.x + self.spacing.x * i as f64,
            self.origin.y + self.spacing.y * j as f64,
            self.origin.z + self.spacing.z * k as f64,
        )
    }

    /// Cartesian coordinate of a linear index, in bohr.
    pub fn point_at(&self, linear: usize) -> Point3<f64> {
        let (i, j, k) = self.multi_index(linear);
        self.point(i, j, k)
    }

    /// Volume element `dV = dx * dy * dz` in bohr^3.
    pub fn cell_volume(&self) -> f64 {
        (self.spacing.x * self.spacing.y * self.spacing.z).abs()
    }

    /// Evaluate `f` at every grid point, returning a linear-order buffer of
    /// length [`SpatialGrid3D::len`].
    ///
    /// The buffer layout is documented in [`SpatialGrid3D::index`].
    pub fn map<F: Fn(f64, f64, f64) -> f64>(&self, f: F) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.len());
        for k in 0..self.counts[2] {
            for j in 0..self.counts[1] {
                for i in 0..self.counts[0] {
                    let p = self.point(i, j, k);
                    out.push(f(p.x, p.y, p.z));
                }
            }
        }
        out
    }

    /// Integrated value `sum(f) * dV` in bohr^3 * (unit of `f`).
    pub fn integrate<F: Fn(f64, f64, f64) -> f64>(&self, f: F) -> f64 {
        self.map(f).iter().sum::<f64>() * self.cell_volume()
    }

    // -----------------------------------------------------------------------
    // Public array order
    // -----------------------------------------------------------------------

    /// Linear index into a buffer laid out in the **public array order**.
    ///
    /// The public order is C order over `(i, j, k)` with **`k` fastest**:
    ///
    /// ```text
    /// public_index(i, j, k) = (i * ny + j) * nz + k
    /// ```
    ///
    /// This is the layout of every array that leaves the workspace: the NumPy
    /// arrays returned by the Python API and the volumetric data consumed by the
    /// renderer both use it. It is the **reverse** of [`SpatialGrid3D::index`],
    /// which is the internal layout with `i` fastest.
    ///
    /// Two layouts exist because they serve different masters: the internal one
    /// keeps `x` contiguous for the row-parallel evaluation in `atoms-orbitals`,
    /// while the public one matches the convention NumPy users expect, so that
    /// `out[i, j, k]` is the value at `(x_i, y_j, z_k)` and `out[:, :, k]` is a
    /// plane of constant `z`.
    ///
    /// The conversion is a relabelling, never a transpose of an existing buffer:
    /// callers fill directly in the order they need. [`Self::map_public`] does
    /// this for generic field functions.
    pub fn index_public(&self, i: usize, j: usize, k: usize) -> usize {
        debug_assert!(i < self.counts[0] && j < self.counts[1] && k < self.counts[2]);
        public_linear_index(i, j, k, self.counts)
    }

    /// The `(i, j, k)` multi-index of a linear index in the **public** order.
    pub fn multi_index_public(&self, linear: usize) -> (usize, usize, usize) {
        let [_nx, ny, nz] = self.counts;
        let i = linear / (ny * nz);
        let rest = linear % (ny * nz);
        let j = rest / nz;
        let k = rest % nz;
        (i, j, k)
    }

    /// Evaluate `f` at every grid point, returning a buffer in the **public**
    /// array order (see [`Self::index_public`]).
    pub fn map_public<F: Fn(f64, f64, f64) -> f64>(&self, f: F) -> Vec<f64> {
        let [nx, ny, nz] = self.counts;
        let mut out = Vec::with_capacity(self.len());
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let p = self.point(i, j, k);
                    out.push(f(p.x, p.y, p.z));
                }
            }
        }
        out
    }

    /// Cartesian coordinate of a linear index in the **public** order, in bohr.
    pub fn point_at_public(&self, linear: usize) -> Point3<f64> {
        let (i, j, k) = self.multi_index_public(linear);
        self.point(i, j, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_grid(n: usize) -> SpatialGrid3D {
        SpatialGrid3D::new(
            Point3::new(-1.0, -1.0, -1.0),
            Vector3::new(0.5, 0.5, 0.5),
            [n, n, n],
        )
        .unwrap()
    }

    #[test]
    fn grid_length_and_spacing() {
        let g = SpatialGrid3D::centred(Point3::origin(), [10.0, 5.0, 2.0], [5, 3, 2]).unwrap();
        assert_eq!(g.len(), 30);
        // x: 10 - (-10) over 4 intervals
        assert!((g.spacing().x - 5.0).abs() < 1e-14);
        assert!((g.spacing().y - 5.0).abs() < 1e-14);
        assert!((g.spacing().z - 4.0).abs() < 1e-14);
    }

    #[test]
    fn centred_grid_outermost_points_hit_extent() {
        let g = SpatialGrid3D::centred(Point3::origin(), [4.0, 4.0, 4.0], [9, 9, 9]).unwrap();
        let first = g.point(0, 0, 0);
        let last = g.point(8, 8, 8);
        assert!((first.x + 4.0).abs() < 1e-14);
        assert!((last.x - 4.0).abs() < 1e-14);
        assert!((g.spacing().x - 1.0).abs() < 1e-14);
    }

    #[test]
    fn centred_grid_with_offset_origin() {
        let g = SpatialGrid3D::centred(Point3::new(1.0, -2.0, 0.5), [1.0, 1.0, 1.0], [3, 3, 3])
            .unwrap();
        let mid = g.point(1, 1, 1);
        assert!((mid.x - 1.0).abs() < 1e-14);
        assert!((mid.y + 2.0).abs() < 1e-14);
        assert!((mid.z - 0.5).abs() < 1e-14);
    }

    #[test]
    fn index_and_multi_index_round_trip() {
        let g = unit_grid(4);
        for k in 0..4 {
            for j in 0..4 {
                for i in 0..4 {
                    let lin = g.index(i, j, k);
                    assert_eq!(g.multi_index(lin), (i, j, k));
                    assert!(lin < g.len());
                }
            }
        }
    }

    #[test]
    fn index_is_unique_and_dense() {
        let g = unit_grid(6);
        let mut seen = vec![false; g.len()];
        for k in 0..6 {
            for j in 0..6 {
                for i in 0..6 {
                    let lin = g.index(i, j, k);
                    assert!(!seen[lin], "duplicate linear index {lin}");
                    seen[lin] = true;
                }
            }
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn point_at_matches_multi_index() {
        let g = unit_grid(3);
        for lin in 0..g.len() {
            let (i, j, k) = g.multi_index(lin);
            let a = g.point_at(lin);
            let b = g.point(i, j, k);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn rejects_degenerate_inputs() {
        assert!(matches!(
            SpatialGrid3D::new(Point3::origin(), Vector3::new(0.1, 0.1, 0.1), [1, 4, 4]),
            Err(GridError::TooFewPoints { axis: 0, .. })
        ));
        assert!(matches!(
            SpatialGrid3D::new(Point3::origin(), Vector3::new(0.0, 0.1, 0.1), [4, 4, 4]),
            Err(GridError::InvalidSpacing { axis: 0 })
        ));
        assert!(matches!(
            SpatialGrid3D::new(
                Point3::new(f64::NAN, 0.0, 0.0),
                Vector3::new(0.1, 0.1, 0.1),
                [4, 4, 4]
            ),
            Err(GridError::InvalidOrigin)
        ));
        assert!(SpatialGrid3D::new(
            Point3::origin(),
            Vector3::new(f64::INFINITY, 0.1, 0.1),
            [4, 4, 4]
        )
        .is_err());
    }

    #[test]
    fn integrate_constant_recovers_volume() {
        // A 2x3x4 bohr box sampled with cell-aligned points: sum*dV overestimates
        // by exactly half a cell per face, so we check the closed form of the
        // discrete sum rather than the continuous volume.
        let g = SpatialGrid3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.5, 0.5, 0.5),
            [5, 7, 9],
        )
        .unwrap();
        let s = g.integrate(|_, _, _| 1.0);
        let expected = g.len() as f64 * g.cell_volume();
        assert!((s - expected).abs() < 1e-12);
    }

    #[test]
    fn public_linear_index_is_the_single_definition_of_the_layout() {
        // Asymmetric on every axis, so a wrong `ny`/`nz` in the formula cannot
        // pass by symmetry.
        let counts = [3usize, 5, 7];
        let g = SpatialGrid3D::new(Point3::origin(), Vector3::new(1.0, 1.0, 1.0), counts).unwrap();

        let total = counts[0] * counts[1] * counts[2];
        let mut seen = vec![false; total];
        for i in 0..counts[0] {
            for j in 0..counts[1] {
                for k in 0..counts[2] {
                    let idx = public_linear_index(i, j, k, counts);
                    assert_eq!(
                        idx,
                        (i * counts[1] + j) * counts[2] + k,
                        "i={i} j={j} k={k}"
                    );
                    // the method must agree with the free function
                    assert_eq!(g.index_public(i, j, k), idx);
                    // the map is a bijection onto 0..total
                    assert!(!seen[idx], "duplicate index {idx}");
                    seen[idx] = true;
                }
            }
        }
        assert!(seen.iter().all(|&b| b), "the map is not onto 0..total");

        // `k` is fastest, so consecutive indices advance in z.
        assert_eq!(public_linear_index(0, 0, 1, counts), 1);
        // then j, then i
        assert_eq!(public_linear_index(0, 1, 0, counts), counts[2]);
        assert_eq!(public_linear_index(1, 0, 0, counts), counts[1] * counts[2]);
        // and the two layouts genuinely differ
        assert_ne!(g.index_public(1, 0, 0), g.index(1, 0, 0));
    }

    #[test]
    fn multi_index_public_inverts_the_public_layout() {
        let counts = [3usize, 5, 7];
        let g = SpatialGrid3D::new(Point3::origin(), Vector3::new(1.0, 1.0, 1.0), counts).unwrap();
        for linear in 0..counts[0] * counts[1] * counts[2] {
            let (i, j, k) = g.multi_index_public(linear);
            assert_eq!(g.index_public(i, j, k), linear);
            assert!(i < counts[0] && j < counts[1] && k < counts[2]);
        }
    }

    #[test]
    fn map_layout_matches_index() {
        let g = SpatialGrid3D::new(Point3::origin(), Vector3::new(1.0, 10.0, 100.0), [3, 3, 3])
            .unwrap();
        let values = g.map(|x, y, z| x + y + z);
        for k in 0..3 {
            for j in 0..3 {
                for i in 0..3 {
                    let p = g.point(i, j, k);
                    let lin = g.index(i, j, k);
                    assert!((values[lin] - (p.x + p.y + p.z)).abs() < 1e-12);
                }
            }
        }
    }

    // -------------------------------------------------------------------
    // Public array order
    // -------------------------------------------------------------------

    /// A deliberately asymmetric grid with pairwise-distinct dimensions, so that
    /// no accidental transpose or axis swap can pass by symmetry.
    fn asymmetric_grid() -> SpatialGrid3D {
        SpatialGrid3D::new(
            Point3::new(-1.0, -2.0, -3.0),
            Vector3::new(0.5, 0.25, 0.125),
            [3, 5, 7],
        )
        .unwrap()
    }

    /// A marker value that encodes a point's own multi-index, giving every grid
    /// point a distinct value: `f(i, j, k) = i*100 + j*10 + k`.
    fn marker(i: usize, j: usize, k: usize) -> f64 {
        (i * 100 + j * 10 + k) as f64
    }

    /// The public order must be a relabelling: reading a publicly-ordered buffer
    /// at `index_public(i, j, k)` must return the value that belongs to
    /// `(x_i, y_j, z_k)`. Asymmetric grid, distinct dimensions, distinct values.
    #[test]
    fn public_order_round_trips_on_an_asymmetric_grid() {
        let g = asymmetric_grid();
        let counts = g.counts();
        assert_eq!(counts, [3, 5, 7]);

        let origin = g.origin();
        let spacing = g.spacing();
        let buf = g.map_public(move |x, y, z| {
            let i = ((x - origin.x) / spacing.x).round() as usize;
            let j = ((y - origin.y) / spacing.y).round() as usize;
            let k = ((z - origin.z) / spacing.z).round() as usize;
            marker(i, j, k)
        });

        assert_eq!(buf.len(), 3 * 5 * 7);
        // every value is distinct, so the layout cannot hide a transposition
        let mut sorted = buf.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted.dedup();
        assert_eq!(sorted.len(), buf.len(), "marker values must be distinct");

        for i in 0..counts[0] {
            for j in 0..counts[1] {
                for k in 0..counts[2] {
                    let lin = g.index_public(i, j, k);
                    assert_eq!(
                        buf[lin],
                        marker(i, j, k),
                        "public order broken at (i={i}, j={j}, k={k})"
                    );
                    assert_eq!(g.multi_index_public(lin), (i, j, k));
                }
            }
        }
    }

    /// The public and internal orders must actually differ on an asymmetric
    /// grid; if they coincided, the distinction would be untested.
    #[test]
    fn public_and_internal_orders_differ_for_asymmetric_grids() {
        let g = asymmetric_grid();
        let mut differ = 0;
        for i in 0..3 {
            for j in 0..5 {
                for k in 0..7 {
                    if g.index(i, j, k) != g.index_public(i, j, k) {
                        differ += 1;
                    }
                }
            }
        }
        assert!(differ > 80, "only {differ} of 105 indices differ");
    }

    #[test]
    fn both_index_maps_are_bijections_onto_the_same_range() {
        let g = asymmetric_grid();
        let mut seen = vec![false; g.len()];
        for i in 0..3 {
            for j in 0..5 {
                for k in 0..7 {
                    let a = g.index(i, j, k);
                    let b = g.index_public(i, j, k);
                    assert!(a < g.len() && b < g.len());
                    seen[a] = true;
                    seen[b] = true;
                }
            }
        }
        assert!(seen.iter().all(|&b| b), "an index was never produced");
    }

    /// `map_public` must produce the same *values*, merely in a different order.
    #[test]
    fn map_public_preserves_the_value_multiset() {
        let g = asymmetric_grid();
        let field = |x: f64, y: f64, z: f64| x * x + 2.0 * y - z;
        let mut a = g.map(field);
        let mut b = g.map_public(field);
        a.sort_by(|p, q| p.partial_cmp(q).unwrap());
        b.sort_by(|p, q| p.partial_cmp(q).unwrap());
        assert_eq!(a, b);
    }

    #[test]
    fn point_at_public_matches_the_multi_index() {
        let g = asymmetric_grid();
        for lin in 0..g.len() {
            let (i, j, k) = g.multi_index_public(lin);
            assert_eq!(g.point_at_public(lin), g.point(i, j, k));
        }
    }

    /// The two layouts must agree for a *symmetric* cubic grid only when the
    /// grid is also indexed symmetrically — they generally do not, which is why
    /// both are documented explicitly.
    #[test]
    fn cube_grids_still_distinguish_the_two_layouts() {
        let g =
            SpatialGrid3D::new(Point3::origin(), Vector3::new(1.0, 1.0, 1.0), [4, 4, 4]).unwrap();
        // interior point: the two orders disagree
        assert_ne!(g.index(1, 2, 3), g.index_public(1, 2, 3));
        // only the origin coincides
        assert_eq!(g.index(0, 0, 0), g.index_public(0, 0, 0));
    }
}
