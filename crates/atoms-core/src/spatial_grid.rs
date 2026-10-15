//! 3D spatial grid for wavefunction evaluation
//! 
//! Defines a regular 3D Cartesian grid with configurable extent and resolution.
//! Supports atomic units (bohr) and other length scales via conversion factors.

use nalgebra::{Point3, Span};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpatialGrid3D {
    pub origin: Point3<f64>,     // Grid origin (x_min, y_min, z_min) in bohr
    pub extent: [f64; 3],        // Grid dimensions (nx, ny, nz) points
    pub resolution: [usize; 3],  // Number of grid points per dimension
}

impl SpatialGrid3D {
    /// Create a new 3D grid with given origin and resolution.
    pub fn new(origin: Point3<f64>, extent: [f64; 3], resolution: [usize; 3]) -> Self {
        Self { origin, extent, resolution }
    }

    /// Create a centered grid symmetric around the origin.
    pub fn centered(
        dimensions: [usize; 3],
        bounds: [f64; 2],  // (r_min, r_max) in bohr
    ) -> Self {
        let span = bounds[1] - bounds[0];
        
        let nx = dimensions[0];
        let ny = dimensions[1];
        let nz = dimensions[2];
        
        // Uniform spacing
        let dx = span / (nx as f64 - 1.0);
        let dy = span / (ny as f64 - 1.0);
        let dz = span / (nz as f64 - 1.0);
        
        // Centered at origin
        let origin_x = -(span / 2.0) + dx * 0.5;
        let origin_y = -(span / 2.0) + dy * 0.5;
        let origin_z = -(span / 2.0) + dz * 0.5;
        
        Self {
            origin: Point3::new(origin_x, origin_y, origin_z),
            extent: [nx, ny, nz],
            resolution: [nx, ny, nz],
        }
    }

    /// Get the total number of grid points
    pub fn size(&self) -> usize {
        self.resolution[0] * self.resolution[1] * self.resolution[2]
    }

    /// Convert extent to SI units (meters) given origin in bohr
    pub fn to_si_origin(&self) -> Point3<f64> {
        use crate::units::conversions;
        let bohr = Point3::new(self.origin.x, self.origin.y, self.origin.z);
        Point3::new(
            bohr.x * conversions::BOHR_TO_SI,
            bohr.y * conversions::BOHR_TO_SI,
            bohr.z * conversions::BOHR_TO_SI,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

    #[test]
    fn test_grid_creation() {
        let grid = SpatialGrid3D::new(
            Point3::new(-10.0, -10.0, -10.0),
            [20.0, 20.0, 20.0],
            [64, 64, 64],
        );
        
        assert_eq!(grid.reservation[0], 64);
        assert_eq!(grid.size(), 64 * 64 * 64);
    }

    #[test]
    fn test_centered_grid() {
        let grid = SpatialGrid3D::centered([128, 128, 128], [-50.0, 50.0]);
        assert_eq!(grid.resolution[0], 128);
        assert!((grid.origin.x).abs() < 1e-10); // Approximately centered
    }
}
