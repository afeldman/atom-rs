//! PyO3 bindings: the `atoms._atoms` extension module.
//!
//! # Design rules enforced here
//!
//! * The binding layer contains **no physics**. Every number comes from
//!   `atoms-core`/`atoms-orbitals`; this module only validates arguments,
//!   converts units, and marshals arrays.
//! * Every invalid argument raises an idiomatic Python exception. No Rust panic
//!   may cross the FFI boundary, so fallible operations return `PyResult` and
//!   errors are mapped to `ValueError`/`TypeError` explicitly.
//! * Large arrays cross the boundary by moving an owned buffer into NumPy
//!   (`IntoPyArray`), not by copying element by element. Input arrays are read
//!   through `PyReadonlyArray1` and never copied.
//! * All physical results are in **Hartree atomic units**, whatever unit was used
//!   to *specify* a grid. Only the grid's own geometry is reported back in the
//!   caller's unit.

use atoms_core::atom::{Atom, AtomicSpecies};
use atoms_core::spatial_grid::{GridError, SpatialGrid3D};
use atoms_core::units::{EnergyUnit, LengthUnit};
use atoms_core::EnergyLevel;
use atoms_io::CalculationMetadata;
use atoms_numerics::sampling::SeededRng;
use atoms_orbitals::{
    expectations as analytic, HydrogenWavefunction, OrbitalSampler, QuantumNumberError,
};
use nalgebra::{Point3, Vector3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyComplex;

use crate::ordering::probability_density_numpy_order;

// ---------------------------------------------------------------------------
// Unit parsing
// ---------------------------------------------------------------------------

fn parse_length_unit(s: &str) -> PyResult<LengthUnit> {
    match s.trim().to_ascii_lowercase().replace(' ', "").as_str() {
        "bohr" | "bohrs" | "a0" | "au" | "atomic" => Ok(LengthUnit::Bohr),
        "angstrom" | "angstroms" | "a" => Ok(LengthUnit::Angstrom),
        "nanometer" | "nanometers" | "nm" => Ok(LengthUnit::Nanometer),
        other => Err(PyValueError::new_err(format!(
            "unknown length unit {other:?}; expected \"bohr\", \"angstrom\" or \"nanometer\""
        ))),
    }
}

fn length_unit_name(u: LengthUnit) -> &'static str {
    match u {
        LengthUnit::Bohr => "bohr",
        LengthUnit::Angstrom => "angstrom",
        LengthUnit::Nanometer => "nanometer",
    }
}

fn parse_energy_unit(s: &str) -> PyResult<EnergyUnit> {
    match s.trim().to_ascii_lowercase().replace(' ', "").as_str() {
        "hartree" | "hartrees" | "ha" | "eh" | "atomic" => Ok(EnergyUnit::Hartree),
        "ev" | "electronvolt" | "electronvolts" => Ok(EnergyUnit::ElectronVolt),
        "kj/mol" | "kjmol" | "kilojoulepermole" => Ok(EnergyUnit::KilojoulePerMole),
        "j" | "joule" | "joules" => Ok(EnergyUnit::Joule),
        other => Err(PyValueError::new_err(format!(
            "unknown energy unit {other:?}; expected \"hartree\", \"ev\", \"kj/mol\" or \"joule\""
        ))),
    }
}

fn qn_error_to_py(e: QuantumNumberError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn grid_error_to_py(e: GridError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Atom
// ---------------------------------------------------------------------------

/// An atomic species placed in space.
///
/// All internal data are in Hartree atomic units; masses are also available in
/// unified atomic mass units through `mass_amu`.
#[pyclass(name = "Atom", module = "atoms", frozen)]
#[derive(Debug, Clone)]
pub struct PyAtom {
    inner: Atom,
}

#[pymethods]
impl PyAtom {
    /// The hydrogen atom (Z = 1, proton mass) at the origin.
    #[staticmethod]
    fn hydrogen() -> Self {
        Self {
            inner: Atom::hydrogen(),
        }
    }

    /// A neutral species by atomic number, for `1 <= z <= 10`.
    ///
    /// Raises `ValueError` outside that range. The table is deliberately small
    /// and auditable; see `atoms.Atom.nominal_configuration`.
    #[staticmethod]
    fn from_z(z: u32) -> PyResult<Self> {
        let species = AtomicSpecies::from_z(z).ok_or_else(|| {
            PyValueError::new_err(format!(
                "no species data for Z = {z}; this build supports 1 <= z <= 10"
            ))
        })?;
        Ok(Self {
            inner: Atom::new(species, Point3::origin()),
        })
    }

    /// An unscreened one-electron (hydrogenic) orbital for this species.
    ///
    /// `Z > 1` gives the unscreened hydrogenic approximation, not a
    /// multi-electron description.
    ///
    /// Raises `ValueError` if the quantum numbers violate `n >= 1`,
    /// `0 <= l < n`, `-l <= m <= l`.
    fn orbital(&self, n: u32, l: u32, m: i32) -> PyResult<PyOrbital> {
        let z = self.inner.z();
        let wf = HydrogenWavefunction::hydrogenic(n, l, m, z).map_err(qn_error_to_py)?;
        Ok(PyOrbital { inner: wf })
    }

    /// IUPAC chemical symbol, e.g. `"H"`.
    #[getter]
    fn symbol(&self) -> String {
        self.inner.species.symbol.clone()
    }

    /// Nuclear charge `Z`.
    #[getter]
    fn z(&self) -> u32 {
        self.inner.z()
    }

    /// Nuclear mass in unified atomic mass units.
    #[getter]
    fn mass_amu(&self) -> f64 {
        self.inner.species.mass_amu()
    }

    /// Nuclear mass in electron masses (the atomic unit of mass).
    #[getter]
    fn mass_electron_masses(&self) -> f64 {
        self.inner.species.mass_electron_masses
    }

    /// Nominal ground-state electron configuration.
    ///
    /// **Descriptive metadata only.** It is not computed, not used by any
    /// calculation, and must not be presented as an electronic-structure result.
    #[getter]
    fn nominal_configuration(&self) -> String {
        self.inner.species.nominal_configuration.clone()
    }

    /// Whether the unscreened one-electron model is exact for this species.
    ///
    /// True only for hydrogen.
    #[getter]
    fn hydrogenic_model_is_exact(&self) -> bool {
        self.inner.species.hydrogenic_model_is_exact()
    }

    /// Nuclear position in bohr, as `(x, y, z)`.
    #[getter]
    fn position(&self) -> (f64, f64, f64) {
        let p = self.inner.position;
        (p.x, p.y, p.z)
    }

    fn __repr__(&self) -> String {
        format!(
            "Atom(symbol={:?}, z={}, mass_amu={:.6})",
            self.symbol(),
            self.z(),
            self.mass_amu()
        )
    }
}

// ---------------------------------------------------------------------------
// Orbital
// ---------------------------------------------------------------------------

/// A hydrogen-like orbital `psi_{nlm}`.
///
/// Convention: `R_{nl}(r) Y_l^m(theta, phi)` in Hartree atomic units, with
/// `theta` the polar angle from `+z` and `phi` the azimuth from `+x`. Complex
/// spherical harmonics include the Condon-Shortley phase, and negative `m` is
/// fully supported (it is not the same as `|m|`).
///
/// Lengths are in bohr, `|psi|^2` in bohr^-3, radial probability density in
/// bohr^-1, angular probability density in sr^-1, energies in hartree unless a
/// unit is requested.
#[pyclass(name = "Orbital", module = "atoms", frozen)]
#[derive(Debug, Clone)]
pub struct PyOrbital {
    inner: HydrogenWavefunction,
}

impl PyOrbital {
    fn wf(&self) -> &HydrogenWavefunction {
        &self.inner
    }

    /// Vectorised helper for the 1-D profile functions.
    fn map_array<'py, F>(
        &self,
        py: Python<'py>,
        r: &Bound<'py, PyAny>,
        f: F,
        what: &str,
    ) -> PyResult<Bound<'py, PyAny>>
    where
        F: Fn(f64) -> f64,
    {
        if let Ok(scalar) = r.extract::<f64>() {
            return Ok(f(scalar).into_pyobject(py)?.into_any());
        }
        let arr = r.cast::<PyArray1<f64>>().map_err(|_| {
            PyTypeError::new_err(format!(
                "{what}: expected a float or a 1-D float64 numpy array, got {}",
                r.get_type()
                    .name()
                    .map(|n| n.to_string())
                    .unwrap_or_default()
            ))
        })?;
        let view = arr.readonly();
        let input = view.as_slice()?;
        let out: Vec<f64> = input.iter().map(|&x| f(x)).collect();
        Ok(out.into_pyarray(py).into_any())
    }
}

#[pymethods]
impl PyOrbital {
    /// Principal quantum number `n`.
    #[getter]
    fn n(&self) -> u32 {
        self.wf().quantum_numbers().n()
    }

    /// Orbital angular momentum quantum number `l`.
    #[getter]
    fn l(&self) -> u32 {
        self.wf().quantum_numbers().l()
    }

    /// Magnetic quantum number `m`, which may be negative.
    #[getter]
    fn m(&self) -> i32 {
        self.wf().quantum_numbers().m()
    }

    /// Nuclear charge of the orbital.
    #[getter]
    fn z(&self) -> u32 {
        self.wf().z()
    }

    /// The validated triple as `(n, l, m)`.
    fn quantum_numbers(&self) -> (u32, u32, i32) {
        let q = self.wf().quantum_numbers();
        (q.n(), q.l(), q.m())
    }

    /// Spectroscopic designation, e.g. `"2p"` or `"2p (m=-1)"`.
    fn designation(&self) -> String {
        self.wf().designation()
    }

    /// Number of radial nodes, `n - l - 1`.
    #[getter]
    fn radial_nodes(&self) -> u32 {
        self.wf().quantum_numbers().radial_nodes()
    }

    /// Number of angular nodes, `l`.
    #[getter]
    fn angular_nodes(&self) -> u32 {
        self.wf().quantum_numbers().angular_nodes()
    }

    /// Total number of nodes, `n - 1`.
    #[getter]
    fn total_nodes(&self) -> u32 {
        self.wf().quantum_numbers().total_nodes()
    }

    /// Analytic energy eigenvalue.
    ///
    /// `unit` is one of `"hartree"` (default), `"ev"`, `"kj/mol"`, `"joule"`.
    /// The value uses the infinite-nuclear-mass convention, in which
    /// `E_n = -Z^2 / (2 n^2)` is exact.
    #[pyo3(signature = (unit = "hartree"))]
    fn energy(&self, unit: &str) -> PyResult<f64> {
        let u = parse_energy_unit(unit)?;
        let e = self.wf().energy();
        Ok(e.in_unit(u))
    }

    /// Radial wavefunction `R_{nl}(r)` in bohr^-3/2, `r` in bohr.
    ///
    /// Accepts a float (returns a float) or a 1-D float64 array (returns a new
    /// 1-D float64 array of the same length).
    fn radial_wavefunction<'py>(
        &self,
        py: Python<'py>,
        r: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.map_array(py, r, |x| self.wf().radial_value(x), "radial_wavefunction")
    }

    /// Radial probability density `P(r) = r^2 |R_{nl}(r)|^2` in bohr^-1.
    ///
    /// This is **not** `|psi|^2`: `P(r) dr` is the probability of finding the
    /// electron at a radius in `[r, r + dr)`, integrated over all angles, and
    /// `P(0) = 0` for every orbital.
    fn radial_probability_density<'py>(
        &self,
        py: Python<'py>,
        r: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.map_array(
            py,
            r,
            |x| self.wf().radial_probability_density(x),
            "radial_probability_density",
        )
    }

    /// Angular probability density `|Y_l^m(theta)|^2` in sr^-1, `theta` in
    /// radians (polar angle from `+z`).
    ///
    /// Independent of `phi`. `int_0^pi |Y|^2 sin(theta) dtheta = 1/(2 pi)`.
    fn angular_probability_density<'py>(
        &self,
        py: Python<'py>,
        theta: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.map_array(
            py,
            theta,
            |t| self.wf().angular_probability_density(t),
            "angular_probability_density",
        )
    }

    /// `psi(x, y, z)` as a Python `complex`; positions in bohr.
    fn value_at<'py>(&self, py: Python<'py>, x: f64, y: f64, z: f64) -> Bound<'py, PyComplex> {
        let v = self.wf().value_cartesian(x, y, z);
        PyComplex::from_doubles(py, v.re, v.im)
    }

    /// Probability density `|psi(x, y, z)|^2` in bohr^-3; positions in bohr.
    fn probability_density_at(&self, x: f64, y: f64, z: f64) -> f64 {
        self.wf().probability_density(x, y, z)
    }

    /// `|psi|^2` sampled on `grid`, as a `float64` array of shape
    /// `(nx, ny, nz)` in C order, in bohr^-3.
    ///
    /// `out[i, j, k]` is the value at the grid point `(x_i, y_j, z_k)`, so
    /// `out[:, :, k]` is a slice of constant `z` and
    /// `out[i, j, k] * grid.volume_element_bohr3` is the probability in that cell.
    fn probability_density<'py>(
        &self,
        py: Python<'py>,
        grid: &PyGrid3D,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let values = probability_density_numpy_order(self.wf(), grid.grid());
        let [nx, ny, nz] = grid.grid().counts();
        let flat = values.into_pyarray(py);
        Ok(flat.reshape([nx, ny, nz])?)
    }

    /// Integrated probability `sum |psi|^2 dV` over `grid`.
    ///
    /// The discrete approximation to `int |psi|^2 dV`. It is deliberately **not**
    /// renormalised, so it can be used to judge whether a grid is adequate.
    fn total_probability(&self, grid: &PyGrid3D) -> f64 {
        let values = probability_density_numpy_order(self.wf(), grid.grid());
        values.iter().sum::<f64>() * grid.grid().cell_volume()
    }

    /// Analytic `<r>` in bohr (Kramers-Pasternack).
    fn expectation_r(&self) -> f64 {
        let q = self.wf().quantum_numbers();
        analytic::scale_to_z(analytic::expectation_r(q.n(), q.l()), 1, self.wf().z())
    }

    /// Analytic `<r^2>` in bohr^2.
    fn expectation_r_squared(&self) -> f64 {
        let q = self.wf().quantum_numbers();
        analytic::scale_to_z(
            analytic::expectation_r_squared(q.n(), q.l()),
            2,
            self.wf().z(),
        )
    }

    /// Analytic `<1/r>` in bohr^-1.
    fn expectation_inverse_r(&self) -> f64 {
        let q = self.wf().quantum_numbers();
        analytic::scale_to_z(analytic::expectation_inverse_r(q.n()), -1, self.wf().z())
    }

    /// Analytic `<1/r^2>` in bohr^-2.
    fn expectation_inverse_r_squared(&self) -> f64 {
        let q = self.wf().quantum_numbers();
        analytic::scale_to_z(
            analytic::expectation_inverse_r_squared(q.n(), q.l()),
            -2,
            self.wf().z(),
        )
    }

    /// Numerical `<r^k>` in bohr^k by quadrature, for `k >= -2`.
    ///
    /// Raises `ValueError` for `k < -2`, where the integral diverges at the
    /// origin for s orbitals.
    fn radial_expectation_value(&self, k: i32) -> PyResult<f64> {
        let v = atoms_orbitals::radial_expectation_value(self.wf(), k);
        if v.is_nan() {
            return Err(PyValueError::new_err(format!(
                "<r^k> with k = {k} diverges at the origin; k >= -2 is required"
            )));
        }
        Ok(v)
    }

    /// Upper bound of the radial sampling domain in bohr, for the sampler that
    /// `atoms.sample_radial` and `atoms.sample_positions` use.
    #[getter]
    fn sampling_radius_max(&self) -> PyResult<f64> {
        let s = OrbitalSampler::new(self.wf()).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(s.radial_max())
    }

    fn __repr__(&self) -> String {
        format!(
            "Orbital(n={}, l={}, m={}, z={}, designation={:?})",
            self.n(),
            self.l(),
            self.m(),
            self.z(),
            self.designation()
        )
    }
}

// ---------------------------------------------------------------------------
// Grid3D
// ---------------------------------------------------------------------------

/// A regular Cartesian grid of sample points.
///
/// `extent` is either a `(lo, hi)` pair, giving a cube `[lo, hi]^3`, or three
/// such pairs for an anisotropic box. Values are interpreted in `unit`; the
/// outermost sample points lie exactly on the extent, so
/// `spacing = (hi - lo) / (n - 1)`.
///
/// **All physical results are returned in Hartree atomic units** (bohr, hartree,
/// bohr^-3), regardless of `unit`. `unit` only affects how the grid's own
/// geometry is specified and reported. `bohr_extent` and `bohr_spacing` are
/// always available for unit-free work.
#[pyclass(name = "Grid3D", module = "atoms", frozen)]
#[derive(Debug, Clone)]
pub struct PyGrid3D {
    grid: SpatialGrid3D,
    unit: LengthUnit,
}

#[pymethods]
impl PyGrid3D {
    #[new]
    #[pyo3(signature = (extent, resolution, unit = "bohr"))]
    fn new(
        extent: &Bound<'_, PyAny>,
        resolution: (usize, usize, usize),
        unit: &str,
    ) -> PyResult<Self> {
        let unit = parse_length_unit(unit)?;

        let pairs: Vec<(f64, f64)> = if let Ok(pair) = extent.extract::<(f64, f64)>() {
            vec![pair; 3]
        } else if let Ok(v) = extent.extract::<Vec<(f64, f64)>>() {
            if v.len() != 3 {
                return Err(PyValueError::new_err(format!(
                    "extent must be (lo, hi) or three (lo, hi) pairs; got {} pairs",
                    v.len()
                )));
            }
            v
        } else {
            return Err(PyValueError::new_err(
                "extent must be (lo, hi) or ((x0,x1),(y0,y1),(z0,z1))",
            ));
        };

        let counts = [resolution.0, resolution.1, resolution.2];
        for (axis, &n) in counts.iter().enumerate() {
            if n < 2 {
                return Err(PyValueError::new_err(format!(
                    "resolution axis {axis} must be at least 2, got {n}"
                )));
            }
        }

        let mut lo_bohr = [0.0_f64; 3];
        let mut spacing = Vector3::zeros();
        for axis in 0..3 {
            let (lo, hi) = pairs[axis];
            if !lo.is_finite() || !hi.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "extent axis {axis} is not finite: ({lo}, {hi})"
                )));
            }
            let lo_b = unit.to_bohr(lo);
            let hi_b = unit.to_bohr(hi);
            if !(hi_b > lo_b) {
                return Err(PyValueError::new_err(format!(
                    "extent axis {axis} must satisfy lo < hi, got ({lo}, {hi})"
                )));
            }
            lo_bohr[axis] = lo_b;
            spacing[axis] = (hi_b - lo_b) / (counts[axis] as f64 - 1.0);
        }

        let grid = SpatialGrid3D::new(
            Point3::new(lo_bohr[0], lo_bohr[1], lo_bohr[2]),
            spacing,
            counts,
        )
        .map_err(grid_error_to_py)?;

        Ok(Self { grid, unit })
    }

    /// The unit used to specify the grid.
    #[getter]
    fn unit(&self) -> &'static str {
        length_unit_name(self.unit)
    }

    /// Number of points per axis, `(nx, ny, nz)`.
    #[getter]
    fn resolution(&self) -> (usize, usize, usize) {
        let c = self.grid.counts();
        (c[0], c[1], c[2])
    }

    /// Shape of an array returned by `Orbital.probability_density`, `(nx, ny, nz)`.
    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        self.resolution()
    }

    /// Grid extent per axis, in `unit`, as `((x0,x1),(y0,y1),(z0,z1))`.
    #[getter]
    fn extent(&self) -> ((f64, f64), (f64, f64), (f64, f64)) {
        let o = self.grid.origin();
        let s = self.grid.spacing();
        let c = self.grid.counts();
        let axis = |origin: f64, spacing: f64, n: usize| {
            (
                self.unit.from_bohr(origin),
                self.unit.from_bohr(origin + spacing * (n as f64 - 1.0)),
            )
        };
        (
            axis(o.x, s.x, c[0]),
            axis(o.y, s.y, c[1]),
            axis(o.z, s.z, c[2]),
        )
    }

    /// Grid extent per axis in bohr.
    #[getter]
    fn bohr_extent(&self) -> ((f64, f64), (f64, f64), (f64, f64)) {
        let o = self.grid.origin();
        let s = self.grid.spacing();
        let c = self.grid.counts();
        let axis =
            |origin: f64, spacing: f64, n: usize| (origin, origin + spacing * (n as f64 - 1.0));
        (
            axis(o.x, s.x, c[0]),
            axis(o.y, s.y, c[1]),
            axis(o.z, s.z, c[2]),
        )
    }

    /// Point spacing `(dx, dy, dz)` in `unit`.
    #[getter]
    fn spacing(&self) -> (f64, f64, f64) {
        let s = self.grid.spacing();
        (
            self.unit.from_bohr(s.x),
            self.unit.from_bohr(s.y),
            self.unit.from_bohr(s.z),
        )
    }

    /// Point spacing `(dx, dy, dz)` in bohr.
    #[getter]
    fn bohr_spacing(&self) -> (f64, f64, f64) {
        let s = self.grid.spacing();
        (s.x, s.y, s.z)
    }

    /// Cell volume `dx * dy * dz` in bohr^3, the weight used by
    /// `Orbital.total_probability`.
    #[getter]
    fn volume_element_bohr3(&self) -> f64 {
        self.grid.cell_volume()
    }

    /// Total number of sample points.
    #[getter]
    fn size(&self) -> usize {
        self.grid.len()
    }

    /// Coordinate axes as three 1-D `float64` arrays, in `unit`.
    ///
    /// `axes()[0][i]` is the `x` coordinate of array index `i`.
    fn axes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let o = self.grid.origin();
        let s = self.grid.spacing();
        let c = self.grid.counts();
        let make = |origin: f64, spacing: f64, n: usize| -> Bound<'py, PyArray1<f64>> {
            let v: Vec<f64> = (0..n)
                .map(|i| self.unit.from_bohr(origin + spacing * i as f64))
                .collect();
            v.into_pyarray(py)
        };
        Ok((
            make(o.x, s.x, c[0]),
            make(o.y, s.y, c[1]),
            make(o.z, s.z, c[2]),
        ))
    }

    fn __repr__(&self) -> String {
        let c = self.grid.counts();
        let e = self.extent();
        format!(
            "Grid3D(extent=(({:.4}, {:.4}), ({:.4}, {:.4}), ({:.4}, {:.4})), \
             resolution=({}, {}, {}), unit={:?})",
            e.0 .0,
            e.0 .1,
            e.1 .0,
            e.1 .1,
            e.2 .0,
            e.2 .1,
            c[0],
            c[1],
            c[2],
            self.unit()
        )
    }
}

impl PyGrid3D {
    fn grid(&self) -> &SpatialGrid3D {
        &self.grid
    }
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

/// Draw `n_samples` radii in bohr from `P(r) = r^2 |R_{nl}|^2`.
///
/// Deterministic: the same `seed` and `n_samples` reproduce the same array on
/// every platform. The returned array is `float64`, shape `(n_samples,)`.
#[pyfunction]
#[pyo3(signature = (orbital, n_samples, seed = 0))]
fn sample_radial<'py>(
    py: Python<'py>,
    orbital: &PyOrbital,
    n_samples: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be positive"));
    }
    let sampler =
        OrbitalSampler::new(orbital.wf()).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut rng = SeededRng::from_seed(seed);
    let mut out = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        out.push(sampler.sample_radius(&mut rng));
    }
    Ok(out.into_pyarray(py))
}

/// Draw `n_samples` Cartesian positions in bohr from `|psi|^2`.
///
/// Deterministic in `seed`. The returned array is `float64`, shape
/// `(n_samples, 3)`, C order, with columns `x`, `y`, `z`.
#[pyfunction]
#[pyo3(signature = (orbital, n_samples, seed = 0))]
fn sample_positions<'py>(
    py: Python<'py>,
    orbital: &PyOrbital,
    n_samples: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be positive"));
    }
    let sampler =
        OrbitalSampler::new(orbital.wf()).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut rng = SeededRng::from_seed(seed);
    let mut out = Vec::with_capacity(n_samples * 3);
    for _ in 0..n_samples {
        let p = sampler.sample_cartesian(&mut rng);
        out.extend_from_slice(&p);
    }
    let flat = out.into_pyarray(py);
    Ok(flat.reshape([n_samples, 3])?)
}

/// Reproducibility record for an exported result, as JSON.
///
/// `parameters` is an optional mapping of string keys to string values.
#[pyfunction]
#[pyo3(signature = (model, units, software_version = None, seed = None, backend = None, notes = None, parameters = None))]
#[allow(clippy::too_many_arguments)]
fn metadata_json(
    model: &str,
    units: &str,
    software_version: Option<&str>,
    seed: Option<u64>,
    backend: Option<&str>,
    notes: Option<&str>,
    parameters: Option<Vec<(String, String)>>,
) -> PyResult<String> {
    let version = software_version.unwrap_or(env!("CARGO_PKG_VERSION"));
    let mut md = CalculationMetadata::new(model, units, version);
    if let Some(p) = parameters {
        for (k, v) in p {
            md = md.with_parameter(k, v);
        }
    }
    if let Some(s) = seed {
        md = md.with_seed(s);
    }
    if let Some(b) = backend {
        md = md.with_backend(b);
    }
    if let Some(n) = notes {
        md = md.with_notes(n);
    }
    md.to_json()
        .map_err(|e| PyValueError::new_err(format!("could not serialise metadata: {e}")))
}

/// Unit conversions. Every function takes and returns a float in the units
/// implied by its name; use NumPy to vectorise.
#[pyfunction]
fn bohr_to_angstrom(v: f64) -> f64 {
    atoms_core::units::conversions::bohr_to_angstrom(v)
}

/// Angstrom -> bohr.
#[pyfunction]
fn angstrom_to_bohr(v: f64) -> f64 {
    atoms_core::units::conversions::angstrom_to_bohr(v)
}

/// Bohr -> nanometre.
#[pyfunction]
fn bohr_to_nanometer(v: f64) -> f64 {
    atoms_core::units::conversions::bohr_to_metre(v) * 1e9
}

/// Nanometre -> bohr.
#[pyfunction]
fn nanometer_to_bohr(v: f64) -> f64 {
    atoms_core::units::conversions::metre_to_bohr(v * 1e-9)
}

/// Hartree -> electronvolt.
#[pyfunction]
fn hartree_to_ev(v: f64) -> f64 {
    atoms_core::units::conversions::hartree_to_ev(v)
}

/// Electronvolt -> hartree.
#[pyfunction]
fn ev_to_hartree(v: f64) -> f64 {
    atoms_core::units::conversions::ev_to_hartree(v)
}

/// Hartree -> kJ/mol.
#[pyfunction]
fn hartree_to_kj_per_mol(v: f64) -> f64 {
    atoms_core::units::conversions::hartree_to_kj_per_mol(v)
}

/// Hartree -> joule.
#[pyfunction]
fn hartree_to_joule(v: f64) -> f64 {
    atoms_core::units::conversions::hartree_to_joule(v)
}

/// Reduced-mass ratio `mu/m_e = M/(M + m_e)` for a nucleus of mass `M` given in
/// electron masses. For a proton, `1836.15267344` gives `0.9994556794247658`.
#[pyfunction]
fn reduced_mass_ratio(nuclear_mass_in_electron_masses: f64) -> f64 {
    atoms_core::hydrogen_energy::reduced_mass_ratio(nuclear_mass_in_electron_masses)
}

/// Energy of level `n` for nuclear charge `z`, in `unit`.
#[pyfunction]
#[pyo3(signature = (n, z, unit = "hartree"))]
fn energy_level(n: u32, z: u32, unit: &str) -> PyResult<f64> {
    let u = parse_energy_unit(unit)?;
    let e: EnergyLevel = EnergyLevel::new(n, z).ok_or_else(|| {
        PyValueError::new_err(format!("invalid quantum numbers: n = {n}, z = {z}"))
    })?;
    Ok(e.in_unit(u))
}

/// Complex spherical harmonic `Y_l^m(theta, phi)`.
///
/// `theta` is the polar angle from `+z` in `[0, pi]`, `phi` the azimuth from
/// `+x` in `[0, 2 pi)`, both in radians. Includes the Condon-Shortley phase;
/// negative `m` obeys `Y_l^{-m} = (-1)^m conj(Y_l^m)`.
///
/// Raises `ValueError` if `|m| > l`.
#[pyfunction]
fn spherical_harmonic(py: Python<'_>, l: u32, m: i32, theta: f64, phi: f64) -> PyResult<Py<PyAny>> {
    if (m.unsigned_abs() as u64) > l as u64 {
        return Err(PyValueError::new_err(format!(
            "spherical harmonic requires -l <= m <= l, got l = {l}, m = {m}"
        )));
    }
    let y = atoms_orbitals::spherical_harmonics::evaluate(l, m, theta, phi);
    Ok(PyComplex::from_doubles(py, y.re, y.im).into_any().unbind())
}

/// `|Y_l^m(theta)|^2` in sr^-1, independent of `phi`.
///
/// `int_0^pi |Y|^2 sin(theta) dtheta = 1 / (2 pi)`.
///
/// Raises `ValueError` if `|m| > l`.
#[pyfunction]
fn spherical_harmonic_abs_squared(l: u32, m: i32, theta: f64) -> PyResult<f64> {
    if (m.unsigned_abs() as u64) > l as u64 {
        return Err(PyValueError::new_err(format!(
            "spherical harmonic requires -l <= m <= l, got l = {l}, m = {m}"
        )));
    }
    Ok(atoms_orbitals::spherical_harmonics::abs_squared(
        l, m, theta,
    ))
}

#[pymodule]
fn _atoms(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyAtom>()?;
    m.add_class::<PyOrbital>()?;
    m.add_class::<PyGrid3D>()?;
    m.add_function(wrap_pyfunction!(sample_radial, m)?)?;
    m.add_function(wrap_pyfunction!(sample_positions, m)?)?;
    m.add_function(wrap_pyfunction!(metadata_json, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_harmonic, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_harmonic_abs_squared, m)?)?;
    m.add_function(wrap_pyfunction!(bohr_to_angstrom, m)?)?;
    m.add_function(wrap_pyfunction!(angstrom_to_bohr, m)?)?;
    m.add_function(wrap_pyfunction!(bohr_to_nanometer, m)?)?;
    m.add_function(wrap_pyfunction!(nanometer_to_bohr, m)?)?;
    m.add_function(wrap_pyfunction!(hartree_to_ev, m)?)?;
    m.add_function(wrap_pyfunction!(ev_to_hartree, m)?)?;
    m.add_function(wrap_pyfunction!(hartree_to_kj_per_mol, m)?)?;
    m.add_function(wrap_pyfunction!(hartree_to_joule, m)?)?;
    m.add_function(wrap_pyfunction!(reduced_mass_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(energy_level, m)?)?;
    Ok(())
}
