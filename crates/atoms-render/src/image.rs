//! Image output.
//!
//! # Channel convention
//!
//! Rendered images are `RGBA8`. The channels carry two different things on
//! purpose:
//!
//! * **RGB** — the display colour, produced by applying a colour map to the
//!   scalar intensity. This is what a human looks at.
//! * **A** — the scalar intensity itself, quantised to `0..=255`, *without* the
//!   colour map.
//!
//! Keeping the intensity in the alpha channel means the numerical validation
//! compares a single channel that is a direct function of the physics, and the
//! colour map cannot mask or introduce a discrepancy. [`RgbaImage::scalar_bytes`]
//! extracts it.
//!
//! # PNG encoding
//!
//! [`RgbaImage::write_png`] writes a real PNG using stored (uncompressed) DEFLATE
//! blocks and the standard CRC-32/Adler-32 checksums, implemented here so that
//! rendered output can be inspected without adding an image dependency. It is a
//! minimal encoder for artefacts and tests, **not** a general-purpose codec:
//! it produces larger files than a compressing encoder and does not support
//! palettes, bit depths other than 8, or interlacing.

use crate::error::RenderError;
use std::io;
use std::path::Path;

/// An 8-bit RGBA image. Row 0 is the **top** row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbaImage {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

impl RgbaImage {
    /// A black, fully transparent image.
    ///
    /// # Errors
    /// Returns [`RenderError::InvalidGeometry`] for a zero-sized image.
    pub fn new(width: usize, height: usize) -> Result<Self, RenderError> {
        if width == 0 || height == 0 {
            return Err(RenderError::InvalidGeometry {
                field: "image dimensions",
            });
        }
        Ok(Self {
            width,
            height,
            pixels: vec![0; width * height * 4],
        })
    }

    /// Wrap a row-major RGBA8 buffer.
    ///
    /// # Errors
    /// Returns [`RenderError::VolumeLength`] if the buffer length does not match
    /// `width * height * 4`.
    pub fn from_pixels(width: usize, height: usize, pixels: Vec<u8>) -> Result<Self, RenderError> {
        if width == 0 || height == 0 {
            return Err(RenderError::InvalidGeometry {
                field: "image dimensions",
            });
        }
        let expected = width * height * 4;
        if pixels.len() != expected {
            return Err(RenderError::VolumeLength {
                expected,
                got: pixels.len(),
            });
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Image width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Row-major RGBA8 bytes.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// The pixel at `(x, y)`, with `(0, 0)` at the top left.
    pub fn pixel(&self, x: usize, y: usize) -> [u8; 4] {
        let o = (y * self.width + x) * 4;
        [
            self.pixels[o],
            self.pixels[o + 1],
            self.pixels[o + 2],
            self.pixels[o + 3],
        ]
    }

    /// Set the pixel at `(x, y)`.
    pub fn set_pixel(&mut self, x: usize, y: usize, rgba: [u8; 4]) {
        let o = (y * self.width + x) * 4;
        self.pixels[o..o + 4].copy_from_slice(&rgba);
    }

    /// The scalar intensity channel, row-major, one byte per pixel.
    ///
    /// This is the channel that is compared between the CPU and GPU paths.
    pub fn scalar_bytes(&self) -> Vec<u8> {
        self.pixels
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| p[3])
            .collect()
    }

    /// Number of pixels differing from `other` by more than `tolerance`, plus the
    /// largest absolute difference. Used for reporting comparisons.
    ///
    /// # Errors
    /// Returns [`RenderError::VolumeLength`] if the images differ in size.
    pub fn compare_scalar(
        &self,
        other: &RgbaImage,
        tolerance: u8,
    ) -> Result<CompareReport, RenderError> {
        if self.width != other.width || self.height != other.height {
            return Err(RenderError::VolumeLength {
                expected: self.width * self.height,
                got: other.width * other.height,
            });
        }
        let a = self.scalar_bytes();
        let b = other.scalar_bytes();
        let mut max_diff = 0u8;
        let mut differing = 0usize;
        let mut beyond = 0usize;
        let mut sum = 0u64;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = x.abs_diff(*y);
            sum += d as u64;
            max_diff = max_diff.max(d);
            if d > 0 {
                differing += 1;
            }
            if d > tolerance {
                beyond += 1;
            }
        }
        Ok(CompareReport {
            pixels: a.len(),
            differing,
            beyond_tolerance: beyond,
            max_difference: max_diff,
            mean_difference: sum as f64 / a.len() as f64,
            tolerance,
        })
    }

    /// Write an 8-bit RGBA PNG.
    ///
    /// # Errors
    /// Propagates I/O errors.
    pub fn write_png(&self, path: impl AsRef<Path>) -> io::Result<()> {
        std::fs::write(path, self.to_png_bytes())
    }

    /// The PNG file contents.
    pub fn to_png_bytes(&self) -> Vec<u8> {
        // Raw scanlines: one filter byte (0 = None) per row, then RGBA.
        let mut raw = Vec::with_capacity(self.height * (1 + self.width * 4));
        for y in 0..self.height {
            raw.push(0u8);
            let start = y * self.width * 4;
            raw.extend_from_slice(&self.pixels[start..start + self.width * 4]);
        }

        let mut png = Vec::new();
        png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);

        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&(self.width as u32).to_be_bytes());
        ihdr.extend_from_slice(&(self.height as u32).to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit, RGBA, no interlace
        push_chunk(&mut png, b"IHDR", &ihdr);

        push_chunk(&mut png, b"IDAT", &zlib_stored(&raw));
        push_chunk(&mut png, b"IEND", &[]);
        png
    }
}

/// Result of comparing two images.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompareReport {
    /// Number of pixels compared.
    pub pixels: usize,
    /// Pixels differing by at least one quantisation step.
    pub differing: usize,
    /// Pixels differing by more than the tolerance.
    pub beyond_tolerance: usize,
    /// Largest absolute difference, in quantisation steps.
    pub max_difference: u8,
    /// Mean absolute difference, in quantisation steps.
    pub mean_difference: f64,
    /// The tolerance that was applied.
    pub tolerance: u8,
}

impl CompareReport {
    /// Whether every pixel was within tolerance.
    pub fn within_tolerance(&self) -> bool {
        self.beyond_tolerance == 0
    }

    /// Fraction of pixels that differ by at least one step.
    pub fn differing_fraction(&self) -> f64 {
        if self.pixels == 0 {
            0.0
        } else {
            self.differing as f64 / self.pixels as f64
        }
    }
}

impl std::fmt::Display for CompareReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} px: max |d| = {}, mean |d| = {:.4}, differing = {} ({:.2}%), \
             beyond tolerance {} = {}",
            self.pixels,
            self.max_difference,
            self.mean_difference,
            self.differing,
            self.differing_fraction() * 100.0,
            self.tolerance,
            self.beyond_tolerance
        )
    }
}

/// The colour map used for display: five viridis anchors, linearly interpolated.
///
/// Identical to the anchors used in `python/examples/orbital_explorer.py`, so the
/// Rust and Python artefacts are directly comparable.
pub const VIRIDIS_ANCHORS: [[f32; 3]; 5] = [
    [0.267004, 0.004874, 0.329415],
    [0.229739, 0.322361, 0.545706],
    [0.127568, 0.566949, 0.550556],
    [0.369214, 0.788888, 0.382914],
    [0.993248, 0.906157, 0.143936],
];

/// Map a normalised intensity in `[0, 1]` to RGB bytes.
///
/// This affects the visible colour only; the numerical validation uses the alpha
/// channel, so a colour-map difference can never be mistaken for a physics
/// difference.
pub fn viridis(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let n = VIRIDIS_ANCHORS.len() - 1;
    let pos = t * n as f32;
    let lo = (pos.floor() as usize).min(n - 1);
    let frac = pos - lo as f32;
    let mut out = [0u8; 3];
    for c in 0..3 {
        let v = VIRIDIS_ANCHORS[lo][c] * (1.0 - frac) + VIRIDIS_ANCHORS[lo + 1][c] * frac;
        out[c] = (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
    out
}

/// Compose a display pixel from a scalar intensity.
pub fn display_pixel(scalar: u8) -> [u8; 4] {
    let rgb = viridis(scalar as f32 / 255.0);
    [rgb[0], rgb[1], rgb[2], scalar]
}

// ---------------------------------------------------------------------------
// Minimal PNG/checksum helpers
// ---------------------------------------------------------------------------

fn push_chunk(out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(tag);
    out.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(tag);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// A zlib stream whose DEFLATE payload uses only stored (uncompressed) blocks.
fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + data.len() / 65535 * 5 + 16);
    out.push(0x78); // CMF: deflate, 32 KiB window
    out.push(0x01); // FLG: no dictionary, check bits set for 0x7801
    let mut offset = 0usize;
    if data.is_empty() {
        out.extend_from_slice(&[0x01, 0x00, 0x00, 0xff, 0xff]);
    }
    while offset < data.len() {
        let len = (data.len() - offset).min(65535);
        let final_block = offset + len == data.len();
        out.push(if final_block { 0x01 } else { 0x00 });
        out.extend_from_slice(&(len as u16).to_le_bytes());
        out.extend_from_slice(&(!(len as u16)).to_le_bytes());
        out.extend_from_slice(&data[offset..offset + len]);
        offset += len;
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_basics() {
        let mut img = RgbaImage::new(4, 3).unwrap();
        assert_eq!(img.width(), 4);
        assert_eq!(img.height(), 3);
        assert_eq!(img.pixels().len(), 48);
        img.set_pixel(1, 2, [1, 2, 3, 4]);
        assert_eq!(img.pixel(1, 2), [1, 2, 3, 4]);
        assert_eq!(img.pixel(0, 0), [0, 0, 0, 0]);
        assert_eq!(img.scalar_bytes().len(), 12);
    }

    #[test]
    fn rejects_bad_dimensions() {
        assert!(RgbaImage::new(0, 5).is_err());
        assert!(RgbaImage::new(5, 0).is_err());
        assert!(RgbaImage::from_pixels(2, 2, vec![0; 15]).is_err());
        assert!(RgbaImage::from_pixels(2, 2, vec![0; 16]).is_ok());
    }

    #[test]
    fn scalar_channel_is_the_alpha_byte() {
        let mut img = RgbaImage::new(2, 2).unwrap();
        img.set_pixel(0, 0, [10, 20, 30, 40]);
        assert_eq!(img.scalar_bytes()[0], 40);
    }

    #[test]
    fn comparison_reports_differences() {
        let mut a = RgbaImage::new(3, 1).unwrap();
        let mut b = RgbaImage::new(3, 1).unwrap();
        a.set_pixel(0, 0, [0, 0, 0, 10]);
        a.set_pixel(1, 0, [0, 0, 0, 100]);
        a.set_pixel(2, 0, [0, 0, 0, 200]);
        b.set_pixel(0, 0, [0, 0, 0, 10]);
        b.set_pixel(1, 0, [0, 0, 0, 101]);
        b.set_pixel(2, 0, [0, 0, 0, 203]);

        // differences are 0, 1 and 3
        let r = a.compare_scalar(&b, 1).unwrap();
        assert_eq!(r.pixels, 3);
        assert_eq!(r.differing, 2);
        assert_eq!(r.max_difference, 3);
        assert_eq!(r.mean_difference, 4.0 / 3.0);
        // one pixel exceeds a tolerance of 1, so the comparison fails
        assert_eq!(r.beyond_tolerance, 1);
        assert!(!r.within_tolerance());
        assert!((r.differing_fraction() - 2.0 / 3.0).abs() < 1e-12);
        assert!(r.to_string().contains("max |d| = 3"));

        // with a tolerance of 3 every pixel is within it
        let relaxed = a.compare_scalar(&b, 3).unwrap();
        assert_eq!(relaxed.beyond_tolerance, 0);
        assert!(relaxed.within_tolerance());

        // a strict comparison rejects any difference
        let strict = a.compare_scalar(&b, 0).unwrap();
        assert_eq!(strict.beyond_tolerance, 2);

        // identical images agree exactly
        let same = a.compare_scalar(&a, 0).unwrap();
        assert_eq!(same.max_difference, 0);
        assert_eq!(same.differing, 0);
        assert!(same.within_tolerance());

        // size mismatch is an error, not a silent truncation
        assert!(a.compare_scalar(&RgbaImage::new(2, 2).unwrap(), 1).is_err());
    }

    #[test]
    fn viridis_endpoints_and_monotonicity_of_the_luminance() {
        assert_eq!(viridis(0.0), [68, 1, 84]);
        assert_eq!(viridis(1.0), [253, 231, 37]);
        // mid-range must be brighter than the low end
        let lum = |c: [u8; 3]| c[0] as u32 + c[1] as u32 + c[2] as u32;
        assert!(lum(viridis(0.5)) > lum(viridis(0.0)));
        assert!(lum(viridis(1.0)) > lum(viridis(0.5)));
    }

    #[test]
    fn display_pixel_preserves_the_scalar_in_alpha() {
        for s in [0u8, 1, 127, 254, 255] {
            assert_eq!(display_pixel(s)[3], s);
        }
    }

    #[test]
    fn checksums_match_known_values() {
        // CRC-32 of "123456789" is 0xCBF43926 (the standard check value)
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        // Adler-32 of "Wikipedia" is 0x11E60398
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }

    #[test]
    fn png_structure_is_well_formed() {
        let img = RgbaImage::from_pixels(2, 2, vec![7; 16]).unwrap();
        let png = img.to_png_bytes();
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        // first chunk is IHDR with length 13
        assert_eq!(&png[8..12], &13u32.to_be_bytes());
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(&png[16..20], &2u32.to_be_bytes()); // width
        assert_eq!(&png[20..24], &2u32.to_be_bytes()); // height
                                                       // ends with an empty IEND chunk
        assert_eq!(&png[png.len() - 8..png.len() - 4], b"IEND");
        assert_eq!(&png[png.len() - 4..], &crc32(b"IEND").to_be_bytes());
        // a single stored DEFLATE block covers 2 rows of 1 + 8 bytes
        assert!(png.len() > 24 + 17);
    }

    #[test]
    fn png_handles_payloads_larger_than_one_deflate_block() {
        // 300 x 300 x 4 bytes of raw scanlines is 360 300 bytes: more than the
        // 65 535-byte stored-block limit, so the chunking path is exercised.
        let img = RgbaImage::new(300, 300).unwrap();
        let png = img.to_png_bytes();
        assert!(png.len() > 360_000);
        assert_eq!(&png[png.len() - 8..png.len() - 4], b"IEND");
    }
}
