//! # atoms-core
//!
//! Physical constants, unit conventions, coordinate transformations, spatial
//! grids, atomic species data, and the hydrogen energy spectrum.
//!
//! This crate is the **physics kernel** and is intentionally free of any
//! dependency on graphics, GPUs, file formats, or Python. It can be used on a
//! headless machine with no graphics hardware.
//!
//! ## Normative conventions
//!
//! * **Units are Hartree atomic units** throughout. Conversions live in
//!   [`units::conversions`]. See [`units`] for the full table.
//! * **Angles** follow the physics convention: `theta` is the polar angle from
//!   `+z`, `phi` is the azimuth from `+x`. See [`coordinates`].
//! * **Grid buffer layout** is C-order over `(i, j, k)` with `i` varying
//!   fastest. See [`spatial_grid::SpatialGrid3D::index`].

pub mod atom;
pub mod coordinates;
pub mod hydrogen_energy;
pub mod spatial_grid;
pub mod units;

pub use atom::{Atom, AtomicSpecies};
pub use coordinates::{cartesian_to_spherical, spherical_to_cartesian, SphericalPoint};
pub use hydrogen_energy::{reduced_mass_ratio, EnergyLevel};
pub use spatial_grid::{public_linear_index, GridError, SpatialGrid3D};
