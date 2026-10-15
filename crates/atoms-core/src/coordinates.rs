//! Coordinate conventions and transformations.
//!
//! # Convention (normative)
//!
//! Spherical coordinates `(r, theta, phi)` are defined in the **physics**
//! convention:
//!
//! * `r >= 0` is the distance from the origin, in bohr.
//! * `theta` is the **polar (colatitude) angle measured from the `+z` axis**,
//!   in `[0, pi]` radians. It is *not* the azimuth.
//! * `phi` is the **azimuthal angle measured in the `x-y` plane from the `+x`
//!   axis towards the `+y` axis**, in `[0, 2*pi)` radians.
//!
//! Transformations:
//! ```text
//! x = r sin(theta) cos(phi)
//! y = r sin(theta) sin(phi)
//! z = r cos(theta)
//!
//! r     = sqrt(x^2 + y^2 + z^2)
//! theta = acos(z / r)          (0 for r = 0)
//! phi   = atan2(y, x)          (0 for x = y = 0)
//! ```
//!
//! This is the convention used by `scipy.special.sph_harm` (with its polar and
//! azimuthal arguments in the corresponding order) and by every standard
//! quantum-mechanics textbook. It is **not** the geography convention
//! (`theta` = azimuth, `phi` = polar) and it is **not** the engineering
//! convention (`theta` = azimuth from `+x`, `phi` = elevation).

use nalgebra::Point3;
use serde::{Deserialize, Serialize};

/// A point in spherical coordinates. `r` is in bohr, angles in radians.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SphericalPoint {
    /// Radial distance, bohr.
    pub r: f64,
    /// Polar angle from `+z`, radians in `[0, pi]`.
    pub theta: f64,
    /// Azimuthal angle from `+x` towards `+y`, radians in `[0, 2*pi)`.
    pub phi: f64,
}

/// Convert Cartesian `(x, y, z)` in bohr to spherical `(r, theta, phi)`.
///
/// The origin maps to `(0, 0, 0)`; `theta` and `phi` are set to `0` there
/// because they are undefined, and every angular function used in this crate is
/// either finite or multiplied by a vanishing radial factor at `r = 0`.
pub fn cartesian_to_spherical(x: f64, y: f64, z: f64) -> SphericalPoint {
    let r = (x * x + y * y + z * z).sqrt();
    if r == 0.0 {
        return SphericalPoint {
            r: 0.0,
            theta: 0.0,
            phi: 0.0,
        };
    }
    let cos_theta = (z / r).clamp(-1.0, 1.0);
    SphericalPoint {
        r,
        theta: cos_theta.acos(),
        phi: y.atan2(x).rem_euclid(std::f64::consts::TAU),
    }
}

/// Convert a Cartesian point in bohr to spherical.
pub fn point_to_spherical(p: &Point3<f64>) -> SphericalPoint {
    cartesian_to_spherical(p.x, p.y, p.z)
}

/// Convert spherical `(r, theta, phi)` to Cartesian `(x, y, z)` in bohr.
pub fn spherical_to_cartesian(r: f64, theta: f64, phi: f64) -> Point3<f64> {
    let st = theta.sin();
    Point3::new(r * st * phi.cos(), r * st * phi.sin(), r * theta.cos())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    fn assert_close(a: f64, b: f64, tol: f64, ctx: &str) {
        assert!((a - b).abs() < tol, "{ctx}: {a} vs {b}");
    }

    #[test]
    fn cardinal_axes_map_to_expected_angles() {
        // +z axis: theta = 0
        let s = cartesian_to_spherical(0.0, 0.0, 2.0);
        assert_close(s.r, 2.0, 1e-15, "r(+z)");
        assert_close(s.theta, 0.0, 1e-15, "theta(+z)");

        // -z axis: theta = pi
        let s = cartesian_to_spherical(0.0, 0.0, -2.0);
        assert_close(s.theta, PI, 1e-15, "theta(-z)");

        // +x axis: theta = pi/2, phi = 0
        let s = cartesian_to_spherical(3.0, 0.0, 0.0);
        assert_close(s.theta, FRAC_PI_2, 1e-15, "theta(+x)");
        assert_close(s.phi, 0.0, 1e-15, "phi(+x)");

        // +y axis: theta = pi/2, phi = pi/2
        let s = cartesian_to_spherical(0.0, 3.0, 0.0);
        assert_close(s.theta, FRAC_PI_2, 1e-15, "theta(+y)");
        assert_close(s.phi, FRAC_PI_2, 1e-15, "phi(+y)");

        // -x axis: phi = pi
        let s = cartesian_to_spherical(-1.0, 0.0, 0.0);
        assert_close(s.phi, PI, 1e-15, "phi(-x)");

        // -y axis: phi = 3pi/2
        let s = cartesian_to_spherical(0.0, -1.0, 0.0);
        assert_close(s.phi, 3.0 * FRAC_PI_2, 1e-15, "phi(-y)");
    }

    #[test]
    fn round_trip_cartesian_spherical() {
        let samples = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 2.0, 3.0),
            (-1.5, 0.25, -4.0),
            (0.3, -0.7, 0.9),
        ];
        for &(x, y, z) in &samples {
            let s = cartesian_to_spherical(x, y, z);
            let p = spherical_to_cartesian(s.r, s.theta, s.phi);
            assert_close(p.x, x, 1e-14, "x");
            assert_close(p.y, y, 1e-14, "y");
            assert_close(p.z, z, 1e-14, "z");
        }
    }

    #[test]
    fn round_trip_spherical_cartesian() {
        let samples = [
            (1.0, 0.0, 0.0),
            (2.0, PI, 0.0),
            (3.0, FRAC_PI_2, FRAC_PI_4),
            (0.5, 2.3, 5.9),
        ];
        for &(r, t, p) in &samples {
            let c = spherical_to_cartesian(r, t, p);
            let s = point_to_spherical(&c);
            assert_close(s.r, r, 1e-14, "r");
            assert_close(s.theta, t, 1e-14, "theta");
            assert_close(s.phi, p, 1e-14, "phi");
        }
    }

    #[test]
    fn origin_is_handled_without_nan() {
        let s = cartesian_to_spherical(0.0, 0.0, 0.0);
        assert_eq!(s.r, 0.0);
        assert_eq!(s.theta, 0.0);
        assert_eq!(s.phi, 0.0);
    }

    #[test]
    fn azimuth_is_normalised_to_zero_two_pi() {
        // atan2 returns negative angles for the third and fourth quadrants;
        // the public API guarantees [0, 2pi).
        let s = cartesian_to_spherical(1.0, -1.0, 0.0);
        assert!((0.0..std::f64::consts::TAU).contains(&s.phi));
        assert_close(s.phi, 7.0 * FRAC_PI_4, 1e-15, "phi");
    }

    #[test]
    fn theta_stays_in_range_for_extreme_arguments() {
        for &z in &[1e-12, -1e-12, 1e6, -1e6] {
            let s = cartesian_to_spherical(1e-9, 1e-9, z);
            assert!((0.0..=PI).contains(&s.theta), "theta={}", s.theta);
        }
    }
}
