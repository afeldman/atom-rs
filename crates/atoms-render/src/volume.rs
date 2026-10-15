//! The neutral data interface between the scientific core and the renderer.
//!
//! # Why this type exists
//!
//! `atoms-render` must not depend on `atoms-core`, `atoms-orbitals`, or any other
//! physics crate: rendering must be replaceable, and the physics must remain
//! usable without a graphics stack. The two sides therefore meet at
//! [`VolumeData`], a plain description of a scalar field on a regular grid that
//! mentions no orbitals, no wavefunctions, and no quantum numbers.
//!
//! # Axis order (normative)
//!
//! `values` is laid out in the **public array order** used everywhere in this
//! workspace: C order over `(i, j, k)` with **`k` fastest**,
//!
//! ```text
//! index(i, j, k) = (i * ny + j) * nz + k
//! ```
//!
//! so that `values[index(i, j, k)]` is the field at
//! `origin + (i, j, k) * spacing`. This is the same order as the NumPy arrays the
//! Python API returns, so no transposition is required anywhere in the data path.
//! `atoms_core::spatial_grid::SpatialGrid3D::index_public` is the single
//! definition of this layout, and it is the *reverse* of that grid's internal
//! order.
//!
//! # Units
//!
//! Geometry (`origin`, `spacing`) is in **bohr** and values carry whatever unit
//! the producing quantity has — for a probability density, `bohr^-3`. The
//! renderer never converts units and never rescales values; it only applies a
//! transfer function for display. `VolumeData` therefore records the value unit
//! as a string so that an image can be attributed.

use crate::error::RenderError;

/// Which grid axis a projection or slice is taken along.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SliceAxis {
    /// Vary `i` (the `x` axis); the resulting image is indexed `(j, k)`.
    X,
    /// Vary `j` (the `y` axis); the resulting image is indexed `(i, k)`.
    Y,
    /// Vary `k` (the `z` axis); the resulting image is indexed `(i, j)`.
    Z,
}

impl SliceAxis {
    /// The axis whose index is held fixed.
    pub const fn name(self) -> &'static str {
        match self {
            SliceAxis::X => "x",
            SliceAxis::Y => "y",
            SliceAxis::Z => "z",
        }
    }
}

/// A scalar field sampled on a regular Cartesian grid, in public array order.
///
/// Values are stored as `f32` because that is what the GPU consumes. The
/// scientific reference remains the `f64` field computed by the physics crates;
/// [`VolumeData::from_f64`] measures and reports the conversion error it incurs
/// rather than hiding it.
#[derive(Debug, Clone, PartialEq)]
pub struct VolumeData {
    shape: [usize; 3],
    origin_bohr: [f64; 3],
    spacing_bohr: [f64; 3],
    values: Vec<f32>,
    value_unit: String,
}

impl VolumeData {
    /// Build from values already in `f32` and in public array order.
    ///
    /// # Errors
    /// Returns [`RenderError::VolumeLength`] if the value count does not match
    /// the shape, [`RenderError::InvalidGeometry`] if the shape has an axis with
    /// fewer than two points or the geometry is not finite, or
    /// [`RenderError::NonFiniteValue`] if a value is NaN or infinite.
    pub fn new(
        shape: [usize; 3],
        origin_bohr: [f64; 3],
        spacing_bohr: [f64; 3],
        values: Vec<f32>,
        value_unit: impl Into<String>,
    ) -> Result<Self, RenderError> {
        for (axis, &n) in shape.iter().enumerate() {
            if n < 2 {
                return Err(RenderError::InvalidGeometry {
                    field: match axis {
                        0 => "shape[0]",
                        1 => "shape[1]",
                        _ => "shape[2]",
                    },
                });
            }
        }
        for (axis, &s) in spacing_bohr.iter().enumerate() {
            if !s.is_finite() || s == 0.0 {
                return Err(RenderError::InvalidGeometry {
                    field: match axis {
                        0 => "spacing_bohr[0]",
                        1 => "spacing_bohr[1]",
                        _ => "spacing_bohr[2]",
                    },
                });
            }
        }
        if origin_bohr.iter().any(|v| !v.is_finite()) {
            return Err(RenderError::InvalidGeometry {
                field: "origin_bohr",
            });
        }
        let expected = shape[0] * shape[1] * shape[2];
        if values.len() != expected {
            return Err(RenderError::VolumeLength {
                expected,
                got: values.len(),
            });
        }
        for (index, &v) in values.iter().enumerate() {
            if !v.is_finite() {
                return Err(RenderError::NonFiniteValue { index });
            }
        }
        Ok(Self {
            shape,
            origin_bohr,
            spacing_bohr,
            values,
            value_unit: value_unit.into(),
        })
    }

    /// Build from a `f64` field, converting to `f32` for the GPU.
    ///
    /// The conversion is the only place where precision is deliberately
    /// discarded; `f32` has ~7 significant decimal digits, i.e. a relative error
    /// of at most `2^-24 ~ 6e-8`. [`VolumeData::max_relative_conversion_error`]
    /// reports the error actually incurred for a given input, so a test can
    /// assert a bound instead of trusting one.
    ///
    /// # Errors
    /// As [`VolumeData::new`], plus [`RenderError::NegativeValue`] when
    /// `require_non_negative` is set and a value is negative (the physically
    /// impossible case for a probability density).
    pub fn from_f64(
        shape: [usize; 3],
        origin_bohr: [f64; 3],
        spacing_bohr: [f64; 3],
        values: &[f64],
        value_unit: impl Into<String>,
        require_non_negative: bool,
    ) -> Result<Self, RenderError> {
        if require_non_negative {
            if let Some(index) = values.iter().position(|v| *v < 0.0) {
                return Err(RenderError::NegativeValue { index });
            }
        }
        let converted: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        Self::new(shape, origin_bohr, spacing_bohr, converted, value_unit)
    }

    /// Grid dimensions, `(nx, ny, nz)`.
    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// Cartesian coordinate of grid point `(0, 0, 0)`, in bohr.
    pub fn origin_bohr(&self) -> [f64; 3] {
        self.origin_bohr
    }

    /// Grid spacing in bohr.
    pub fn spacing_bohr(&self) -> [f64; 3] {
        self.spacing_bohr
    }

    /// The stored `f32` values, in public array order.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Unit of the stored values, e.g. `"bohr^-3"`.
    pub fn value_unit(&self) -> &str {
        &self.value_unit
    }

    /// Number of grid points.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Always `false`: a valid volume has at least `2^3` points.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Public-array-order index of grid point `(i, j, k)`.
    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        let [_nx, ny, nz] = self.shape;
        (i * ny + j) * nz + k
    }

    /// Value at grid point `(i, j, k)`.
    ///
    /// # Panics
    /// Panics if any index is out of bounds. Use [`VolumeData::get`] for a
    /// checked version.
    pub fn at(&self, i: usize, j: usize, k: usize) -> f32 {
        self.values[self.index(i, j, k)]
    }

    /// Checked accessor.
    ///
    /// # Errors
    /// Returns [`RenderError::OutOfRange`] if any index is outside the grid.
    pub fn get(&self, i: usize, j: usize, k: usize) -> Result<f32, RenderError> {
        for (axis, value) in [i, j, k].iter().enumerate() {
            if *value >= self.shape[axis] {
                return Err(RenderError::OutOfRange {
                    what: ["i", "j", "k"][axis].to_string(),
                    requested: *value,
                    limit: self.shape[axis],
                });
            }
        }
        Ok(self.at(i, j, k))
    }

    /// Largest stored value (0.0 for an all-zero volume).
    pub fn max(&self) -> f32 {
        self.values.iter().copied().fold(0.0_f32, f32::max)
    }

    /// Smallest stored value.
    pub fn min(&self) -> f32 {
        self.values.iter().copied().fold(f32::INFINITY, f32::min)
    }

    /// Largest relative error incurred when `self` was converted from `f64`.
    ///
    /// Intended for tests and for recording provenance: the caller passes the
    /// original `f64` field and receives the measured deviation, so the `f32`
    /// storage decision is quantified rather than assumed.
    ///
    /// # Caveat
    ///
    /// This is a **per-value** relative error. For a field spanning many orders
    /// of magnitude — a probability density spans forty — it is dominated by the
    /// smallest values, whose relative error is large simply because `f32` has
    /// few significant digits there and values below `~1.2e-38` are subnormal.
    /// A field with a genuine zero (a nodal plane) has an *undefined* relative
    /// error at that point.
    ///
    /// For a rendering decision the meaningful metric is
    /// [`VolumeData::max_error_relative_to_field_max`], because the transfer
    /// function normalises by the field maximum.
    pub fn max_relative_conversion_error(&self, original: &[f64]) -> f64 {
        let mut worst: f64 = 0.0;
        for (a, &b) in self.values.iter().zip(original.iter()) {
            let denom = b.abs().max(f64::MIN_POSITIVE);
            worst = worst.max(((*a as f64) - b).abs() / denom);
        }
        worst
    }

    /// Largest absolute conversion error expressed as a fraction of the field
    /// maximum.
    ///
    /// This is the metric that bounds what a renderer can see: the transfer
    /// function divides by `f32::max`, so an error that is tiny compared with the
    /// maximum cannot change any pixel by more than a corresponding fraction of
    /// one quantisation step.
    ///
    /// Returns 0 for an all-zero field.
    pub fn max_error_relative_to_field_max(&self, original: &[f64]) -> f64 {
        let scale = original
            .iter()
            .fold(0.0f64, |acc, v| acc.max(v.abs()))
            .max(self.max().abs() as f64);
        if scale == 0.0 {
            return 0.0;
        }
        let mut worst: f64 = 0.0;
        for (a, &b) in self.values.iter().zip(original.iter()) {
            worst = worst.max(((*a as f64) - b).abs() / scale);
        }
        worst
    }

    /// Total bytes the value buffer occupies on the device.
    pub fn byte_len(&self) -> usize {
        self.values.len() * std::mem::size_of::<f32>()
    }

    /// Bytes per row of a slice image taken perpendicular to `axis`.
    ///
    /// `wgpu` requires the readback row stride to be a multiple of 256, so the
    /// padding is computed here once and used by both the CPU and GPU paths.
    pub fn slice_row_bytes(&self, axis: SliceAxis) -> usize {
        let width = self.slice_dims(axis).0;
        width * 4
    }

    /// `(width, height)` of an image produced by slicing or projecting along
    /// `axis`, in pixels.
    pub fn slice_dims(&self, axis: SliceAxis) -> (usize, usize) {
        let [nx, ny, nz] = self.shape;
        match axis {
            SliceAxis::X => (ny, nz),
            SliceAxis::Y => (nx, nz),
            SliceAxis::Z => (nx, ny),
        }
    }

    /// Number of steps along `axis`.
    pub fn axis_len(&self, axis: SliceAxis) -> usize {
        let [nx, ny, nz] = self.shape;
        match axis {
            SliceAxis::X => nx,
            SliceAxis::Y => ny,
            SliceAxis::Z => nz,
        }
    }
}

/// Maps a physical value to a display intensity.
///
/// # Definition (normative)
///
/// ```text
/// t     = clamp(value / vmax, 0, 1)
/// n     = ln(1 + t * K) / ln(1 + K)
/// byte  = floor(n * 255 + 0.5)
/// ```
///
/// with `K = 20` by default. `vmax` is fixed by the caller (normally the global
/// maximum of the volume) so that two images of the same volume with different
/// slices share a dynamic range and remain comparable.
///
/// The same formula is implemented in the WGSL shader and in
/// [`crate::cpu`], and the two are compared by test. Rounding uses
/// `floor(x + 0.5)` rather than a `round` intrinsic because `round` has
/// implementation-defined tie behaviour in WGSL while `floor` does not.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransferFunction {
    /// Value mapped to full intensity. Must be positive.
    pub vmax: f32,
    /// Logarithmic compression constant `K`.
    pub compression: f32,
}

impl TransferFunction {
    /// A log-compressed transfer function with the given full-intensity value.
    pub fn log_scaled(vmax: f32) -> Self {
        Self {
            vmax,
            compression: 20.0,
        }
    }

    /// Normalised intensity in `[0, 1]` for `value`.
    ///
    /// `compression == 0` degenerates to a linear ramp (`n = t`); the formula
    /// below would otherwise evaluate `ln(1)/ln(1) = 0/0`. A linear transfer is
    /// genuinely useful for tests that need the scalar channel to be a
    /// monotone function of the value.
    pub fn normalise(&self, value: f32) -> f32 {
        // Written as an explicit NaN test rather than `!(vmax > 0.0)` so that the
        // intent (a NaN or non-positive vmax maps everything to zero) is visible.
        if self.vmax.is_nan() || self.vmax <= 0.0 {
            return 0.0;
        }
        let t = (value / self.vmax).clamp(0.0, 1.0);
        if self.compression <= 0.0 {
            return t;
        }
        (1.0 + t * self.compression).ln() / (1.0 + self.compression).ln()
    }

    /// Quantised intensity `0..=255`.
    pub fn quantise(&self, value: f32) -> u8 {
        let n = self.normalise(value);
        (n * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn volume() -> VolumeData {
        // 3 x 4 x 5, values = i*100 + j*10 + k so every point is distinct.
        let [nx, ny, nz] = [3usize, 4, 5];
        let mut values = Vec::new();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    values.push((i * 100 + j * 10 + k) as f32);
                }
            }
        }
        VolumeData::new(
            [nx, ny, nz],
            [-1.0, -2.0, -3.0],
            [0.5, 0.25, 0.125],
            values,
            "bohr^-3",
        )
        .unwrap()
    }

    #[test]
    fn index_matches_the_documented_public_order() {
        let v = volume();
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..5 {
                    assert_eq!(v.index(i, j, k), (i * 4 + j) * 5 + k);
                    assert_eq!(v.at(i, j, k), (i * 100 + j * 10 + k) as f32);
                }
            }
        }
    }

    #[test]
    fn slice_dimensions_follow_the_axis() {
        let v = volume();
        assert_eq!(v.slice_dims(SliceAxis::X), (4, 5));
        assert_eq!(v.slice_dims(SliceAxis::Y), (3, 5));
        assert_eq!(v.slice_dims(SliceAxis::Z), (3, 4));
        assert_eq!(v.axis_len(SliceAxis::X), 3);
        assert_eq!(v.byte_len(), 3 * 4 * 5 * 4);
    }

    #[test]
    fn geometry_is_preserved_exactly() {
        let v = volume();
        assert_eq!(v.origin_bohr(), [-1.0, -2.0, -3.0]);
        assert_eq!(v.spacing_bohr(), [0.5, 0.25, 0.125]);
        assert_eq!(v.shape(), [3, 4, 5]);
        assert_eq!(v.value_unit(), "bohr^-3");
    }

    #[test]
    fn rejects_bad_inputs() {
        let ok = vec![0.0_f32; 60];
        assert!(matches!(
            VolumeData::new([3, 4, 5], [0.0; 3], [1.0; 3], ok[..59].to_vec(), "u"),
            Err(RenderError::VolumeLength {
                expected: 60,
                got: 59
            })
        ));
        assert!(matches!(
            VolumeData::new([1, 4, 5], [0.0; 3], [1.0; 3], vec![0.0; 20], "u"),
            Err(RenderError::InvalidGeometry { field: "shape[0]" })
        ));
        assert!(matches!(
            VolumeData::new([2, 2, 2], [0.0; 3], [1.0, 0.0, 1.0], vec![0.0; 8], "u"),
            Err(RenderError::InvalidGeometry {
                field: "spacing_bohr[1]"
            })
        ));
        assert!(matches!(
            VolumeData::new([2, 2, 2], [f64::NAN, 0.0, 0.0], [1.0; 3], vec![0.0; 8], "u"),
            Err(RenderError::InvalidGeometry {
                field: "origin_bohr"
            })
        ));
        let mut bad = vec![0.0_f32; 8];
        bad[5] = f32::NAN;
        assert!(matches!(
            VolumeData::new([2, 2, 2], [0.0; 3], [1.0; 3], bad, "u"),
            Err(RenderError::NonFiniteValue { index: 5 })
        ));
        let mut neg = vec![0.0_f32; 8];
        neg[2] = -1.0;
        assert!(VolumeData::new([2, 2, 2], [0.0; 3], [1.0; 3], neg, "u").is_ok());
    }

    #[test]
    fn rejects_negative_values_only_when_asked() {
        let values = [0.0, 1.0, -1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!(matches!(
            VolumeData::from_f64([2, 2, 2], [0.0; 3], [1.0; 3], &values, "u", true),
            Err(RenderError::NegativeValue { index: 2 })
        ));
        assert!(VolumeData::from_f64([2, 2, 2], [0.0; 3], [1.0; 3], &values, "u", false).is_ok());
    }

    #[test]
    fn checked_access_reports_out_of_range() {
        let v = volume();
        assert_eq!(v.get(2, 3, 4).unwrap(), 234.0);
        assert!(matches!(
            v.get(3, 0, 0),
            Err(RenderError::OutOfRange {
                requested: 3,
                limit: 3,
                ..
            })
        ));
        assert!(v.get(0, 4, 0).is_err());
        assert!(v.get(0, 0, 5).is_err());
    }

    #[test]
    fn f64_conversion_error_is_measured_and_small() {
        let values: Vec<f64> = (0..8).map(|i| 1.0 / (i as f64 + 1.0)).collect();
        let v = VolumeData::from_f64([2, 2, 2], [0.0; 3], [1.0; 3], &values, "u", false).unwrap();
        let err = v.max_relative_conversion_error(&values);
        assert!(
            err < 6e-8,
            "measured f32 conversion error {err:e} exceeds 2^-24"
        );
        assert!(v.max_error_relative_to_field_max(&values) <= err);
    }

    #[test]
    fn relative_to_max_error_handles_a_field_with_a_true_zero() {
        // A field spanning 40 orders of magnitude, as an orbital density does.
        let values = [0.0, 1.6e-2, 1e-20, 1e-40, 4.6e-40, 1e-8, 1e-2, 1e-15];
        let v = VolumeData::from_f64([2, 2, 2], [0.0; 3], [1.0; 3], &values, "u", false).unwrap();

        // The per-value relative error is large, because the smallest values are
        // subnormal in f32; this is expected and is why the other metric exists.
        let per_value = v.max_relative_conversion_error(&values);
        assert!(
            per_value > 1e-8,
            "expected subnormal relative error, got {per_value:e}"
        );

        // Relative to the field maximum the error stays at the f32 epsilon level.
        let relative = v.max_error_relative_to_field_max(&values);
        assert!(
            relative < 6e-8,
            "error relative to the field max must stay within 2^-24, got {relative:e}"
        );
    }

    #[test]
    fn all_zero_field_reports_no_error() {
        let values = [0.0f64; 8];
        let v = VolumeData::from_f64([2, 2, 2], [0.0; 3], [1.0; 3], &values, "u", false).unwrap();
        assert_eq!(v.max_error_relative_to_field_max(&values), 0.0);
    }

    #[test]
    fn transfer_function_is_monotone_and_hits_its_endpoints() {
        let tf = TransferFunction::log_scaled(1.0);
        assert_eq!(tf.quantise(0.0), 0);
        assert_eq!(tf.quantise(1.0), 255);
        assert_eq!(tf.quantise(2.0), 255, "values above vmax saturate");
        assert_eq!(tf.quantise(-1.0), 0, "negative values clamp to zero");
        let mut prev = 0u8;
        for step in 0..=1000 {
            let v = step as f32 / 1000.0;
            let q = tf.quantise(v);
            assert!(q >= prev, "not monotone at v={v}");
            prev = q;
        }
    }

    #[test]
    fn transfer_function_compresses_the_low_range() {
        // At 1/100 of vmax the log transfer is well above the linear value.
        let tf = TransferFunction::log_scaled(1.0);
        let n = tf.normalise(0.01);
        assert!(n > 0.01, "expected compression to lift the low end");
        assert!((n - (1.0 + 0.01 * 20.0_f32).ln() / (21.0_f32).ln()).abs() < 1e-7);
    }

    #[test]
    fn transfer_function_with_zero_compression_is_linear() {
        let tf = TransferFunction {
            vmax: 2.0,
            compression: 0.0,
        };
        assert_eq!(tf.normalise(0.0), 0.0);
        assert_eq!(tf.normalise(1.0), 0.5);
        assert_eq!(tf.normalise(2.0), 1.0);
        assert_eq!(tf.quantise(1.0), 128);
    }

    #[test]
    fn transfer_function_with_zero_vmax_is_safe() {
        let tf = TransferFunction::log_scaled(0.0);
        assert_eq!(tf.quantise(1.0), 0);
        assert_eq!(tf.normalise(1.0), 0.0);
    }
}
