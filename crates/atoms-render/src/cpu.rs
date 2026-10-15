//! The CPU reference renderer.
//!
//! This is the **reference implementation** for everything the GPU path does, and
//! the fallback used when no adapter is available. It is always compiled, has no
//! dependencies, and runs headless.
//!
//! The GPU path is validated against this module, not against an image that
//! merely looks plausible.
//!
//! # Image orientation (normative)
//!
//! For a slice or projection along axis `a`, the image uses the two remaining
//! axes in ascending order, with the later axis vertical and **increasing
//! upwards** (row 0 of the image is the largest coordinate):
//!
//! | `axis` | image width | image height | pixel `(px, py)` shows |
//! |---|---|---|---|
//! | `X` | `ny` | `nz` | `(i = fixed, j = px, k = nz-1-py)` |
//! | `Y` | `nx` | `nz` | `(i = px, j = fixed, k = nz-1-py)` |
//! | `Z` | `nx` | `ny` | `(i = px, j = ny-1-py, k = fixed)` |
//!
//! So a `Z` slice is an `x`-`y` image with `+y` up, a `Y` slice is `x`-`z` with
//! `+z` up, and an `X` slice is `y`-`z` with `+z` up. The GPU shader implements
//! the same mapping, and a test asserts the two agree pixel for pixel.
//!
//! Values are quantised with [`TransferFunction`] and the display colour comes
//! from the shared colour map; the scalar intensity is preserved in the alpha
//! channel (see [`crate::image`]).

use crate::error::RenderError;
use crate::image::{display_pixel, RgbaImage};
use crate::volume::{SliceAxis, TransferFunction, VolumeData};

/// The two image axes for a projection along `axis`, as `(horizontal, vertical)`
/// indices into `(i, j, k)`.
pub const fn image_axes(axis: SliceAxis) -> (usize, usize) {
    match axis {
        SliceAxis::X => (1, 2),
        SliceAxis::Y => (0, 2),
        SliceAxis::Z => (0, 1),
    }
}

/// Map image coordinates back to a grid multi-index.
fn grid_index(
    axis: SliceAxis,
    fixed: usize,
    px: usize,
    py: usize,
    height: usize,
) -> (usize, usize, usize) {
    let v = height - 1 - py;
    match axis {
        SliceAxis::X => (fixed, px, v),
        SliceAxis::Y => (px, fixed, v),
        SliceAxis::Z => (px, v, fixed),
    }
}

/// Render the slice of `volume` at coordinate `index` along `axis`.
///
/// # Errors
/// Returns [`RenderError::OutOfRange`] if `index` is outside the axis.
pub fn render_slice(
    volume: &VolumeData,
    axis: SliceAxis,
    index: usize,
    tf: &TransferFunction,
) -> Result<RgbaImage, RenderError> {
    let limit = volume.axis_len(axis);
    if index >= limit {
        return Err(RenderError::OutOfRange {
            what: format!("slice index along {}", axis.name()),
            requested: index,
            limit,
        });
    }
    let (width, height) = volume.slice_dims(axis);
    let mut img = RgbaImage::new(width, height)?;
    for py in 0..height {
        for px in 0..width {
            let (i, j, k) = grid_index(axis, index, px, py, height);
            let value = volume.at(i, j, k);
            img.set_pixel(px, py, display_pixel(tf.quantise(value)));
        }
    }
    Ok(img)
}

/// Render a maximum-intensity projection along `axis`, as seen from `+axis`.
///
/// The value at each pixel is the maximum of the volume along the projection
/// direction. For a non-negative field such as a probability density this is the
/// bright "what is the densest thing on this line of sight" image.
///
/// # Errors
/// Currently infallible apart from image allocation.
pub fn render_mip(
    volume: &VolumeData,
    axis: SliceAxis,
    tf: &TransferFunction,
) -> Result<RgbaImage, RenderError> {
    let (width, height) = volume.slice_dims(axis);
    let depth = volume.axis_len(axis);
    let mut img = RgbaImage::new(width, height)?;
    for py in 0..height {
        for px in 0..width {
            let mut best = f32::NEG_INFINITY;
            for d in 0..depth {
                let (i, j, k) = grid_index(axis, d, px, py, height);
                best = best.max(volume.at(i, j, k));
            }
            img.set_pixel(px, py, display_pixel(tf.quantise(best)));
        }
    }
    Ok(img)
}

/// Render a slice using a transfer function scaled to the volume's maximum.
///
/// # Errors
/// As [`render_slice`].
pub fn render_slice_auto(
    volume: &VolumeData,
    axis: SliceAxis,
    index: usize,
) -> Result<RgbaImage, RenderError> {
    let tf = TransferFunction::log_scaled(volume.max());
    render_slice(volume, axis, index, &tf)
}

/// Render a maximum-intensity projection using a transfer function scaled to the
/// volume's maximum.
///
/// # Errors
/// As [`render_mip`].
pub fn render_mip_auto(volume: &VolumeData, axis: SliceAxis) -> Result<RgbaImage, RenderError> {
    let tf = TransferFunction::log_scaled(volume.max());
    render_mip(volume, axis, &tf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3 x 4 x 5 with `value(i, j, k) = i*100 + j*10 + k`, so every point is
    /// distinct and the mapping can be verified exactly.
    fn volume() -> VolumeData {
        let [nx, ny, nz] = [3usize, 4, 5];
        let mut values = Vec::new();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    values.push((i * 100 + j * 10 + k) as f32);
                }
            }
        }
        VolumeData::new([nx, ny, nz], [0.0; 3], [1.0; 3], values, "u").unwrap()
    }

    /// A linear transfer function normalised to the volume's maximum, so the
    /// scalar channel is a monotone function of the value and orientation can be
    /// asserted exactly.
    fn linear_tf() -> TransferFunction {
        TransferFunction {
            vmax: 234.0,
            compression: 0.0,
        }
    }

    #[test]
    fn image_axes_follow_the_documented_rule() {
        assert_eq!(image_axes(SliceAxis::X), (1, 2));
        assert_eq!(image_axes(SliceAxis::Y), (0, 2));
        assert_eq!(image_axes(SliceAxis::Z), (0, 1));
    }

    #[test]
    fn z_slice_orientation_is_x_horizontal_and_y_upwards() {
        let v = volume();
        let tf = linear_tf();
        let img = render_slice(&v, SliceAxis::Z, 2, &tf).unwrap();
        assert_eq!((img.width(), img.height()), (3, 4));

        // pixel (px, py) is (i = px, j = 3 - py, k = 2)
        for py in 0..4 {
            for px in 0..3 {
                let j = 3 - py;
                let expected_value = (px * 100 + j * 10 + 2) as f32;
                let expected = (expected_value / 234.0 * 255.0 + 0.5).floor() as u8;
                assert_eq!(
                    img.pixel(px, py)[3],
                    expected,
                    "pixel ({px}, {py}) should be (i={px}, j={j}, k=2)"
                );
            }
        }
        // top row is the largest j
        assert!(img.pixel(0, 0)[3] > img.pixel(0, 3)[3]);
    }

    #[test]
    fn y_slice_is_x_horizontal_and_z_upwards() {
        let v = volume();
        let tf = linear_tf();
        let img = render_slice(&v, SliceAxis::Y, 3, &tf).unwrap();
        assert_eq!((img.width(), img.height()), (3, 5));
        for py in 0..5 {
            for px in 0..3 {
                let k = 4 - py;
                let expected = (((px * 100 + 30 + k) as f32) / 234.0 * 255.0 + 0.5).floor() as u8;
                assert_eq!(img.pixel(px, py)[3], expected, "pixel ({px}, {py})");
            }
        }
    }

    #[test]
    fn x_slice_is_y_horizontal_and_z_upwards() {
        let v = volume();
        let tf = linear_tf();
        let img = render_slice(&v, SliceAxis::X, 1, &tf).unwrap();
        assert_eq!((img.width(), img.height()), (4, 5));
        for py in 0..5 {
            for px in 0..4 {
                let k = 4 - py;
                let expected = (((100 + px * 10 + k) as f32) / 234.0 * 255.0 + 0.5).floor() as u8;
                assert_eq!(img.pixel(px, py)[3], expected, "pixel ({px}, {py})");
            }
        }
    }

    #[test]
    fn slice_index_is_range_checked() {
        let v = volume();
        let tf = linear_tf();
        assert!(render_slice(&v, SliceAxis::X, 3, &tf).is_err());
        assert!(render_slice(&v, SliceAxis::Y, 4, &tf).is_err());
        assert!(render_slice(&v, SliceAxis::Z, 5, &tf).is_err());
        assert!(render_slice(&v, SliceAxis::Z, 4, &tf).is_ok());
        match render_slice(&v, SliceAxis::Z, 9, &tf) {
            Err(RenderError::OutOfRange {
                requested, limit, ..
            }) => {
                assert_eq!((requested, limit), (9, 5));
            }
            other => panic!("expected OutOfRange, got {other:?}"),
        }
    }

    #[test]
    fn mip_takes_the_maximum_along_the_axis() {
        let v = volume();
        let tf = linear_tf();
        // MIP along Z: pixel (i, j) holds max over k of i*100 + j*10 + k = i*100 + j*10 + 4
        let img = render_mip(&v, SliceAxis::Z, &tf).unwrap();
        assert_eq!((img.width(), img.height()), (3, 4));
        for py in 0..4 {
            for px in 0..3 {
                let j = 3 - py;
                let expected_value = (px * 100 + j * 10 + 4) as f32;
                let expected = (expected_value / 234.0 * 255.0 + 0.5).floor() as u8;
                assert_eq!(img.pixel(px, py)[3], expected, "MIP-Z pixel ({px}, {py})");
            }
        }

        // MIP along Y: max over j of i*100 + j*10 + k = i*100 + 30 + k
        let img = render_mip(&v, SliceAxis::Y, &tf).unwrap();
        assert_eq!((img.width(), img.height()), (3, 5));
        for py in 0..5 {
            for px in 0..3 {
                let k = 4 - py;
                let expected_value = (px * 100 + 30 + k) as f32;
                let expected = (expected_value / 234.0 * 255.0 + 0.5).floor() as u8;
                assert_eq!(img.pixel(px, py)[3], expected, "MIP-Y pixel ({px}, {py})");
            }
        }
    }

    #[test]
    fn mip_is_at_least_every_slice() {
        let v = volume();
        for axis in [SliceAxis::X, SliceAxis::Y, SliceAxis::Z] {
            let mip = render_mip_auto(&v, axis).unwrap();
            let mip_scalars = mip.scalar_bytes();
            let depth = v.axis_len(axis);
            for index in 0..depth {
                let slice = render_slice_auto(&v, axis, index).unwrap();
                // The transfer function is monotone and both images use the same
                // vmax, so the projected value can never quantise below a slice.
                for (s, m) in slice.scalar_bytes().iter().zip(mip_scalars.iter()) {
                    assert!(
                        *m >= *s,
                        "MIP below a slice value for axis {axis:?} at index {index}: \
                         {m} < {s}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_uniform_volume_gives_a_uniform_image() {
        let values = vec![0.5_f32; 2 * 2 * 2];
        let v = VolumeData::new([2, 2, 2], [0.0; 3], [1.0; 3], values, "u").unwrap();
        let img = render_slice_auto(&v, SliceAxis::Z, 0).unwrap();
        let first = img.pixel(0, 0);
        for y in 0..img.height() {
            for x in 0..img.width() {
                assert_eq!(img.pixel(x, y), first);
            }
        }
        assert!(first[3] > 0, "a non-zero volume must not render as black");
    }

    #[test]
    fn all_zero_volume_renders_black() {
        let v = VolumeData::new([2, 2, 2], [0.0; 3], [1.0; 3], vec![0.0; 8], "u").unwrap();
        let img = render_slice_auto(&v, SliceAxis::Z, 0).unwrap();
        assert!(img.scalar_bytes().iter().all(|&b| b == 0));
    }
}
