//! # atoms-io (partially implemented)
//!
//! Documented serialisation and export interfaces.
//!
//! **Status: JSON metadata implemented in M1; XYZ and NumPy/HDF5 scheduled for
//! M2/M3.** File formats are deliberately *not* part of the physics core: the
//! physics crates never depend on this crate.
//!
//! ## Implemented here (M1)
//!
//! * [`CalculationMetadata`] — the reproducibility record that must accompany
//!   every exported result. Its JSON serialisation is stable and tested for
//!   round-trip integrity.
//!
//! ## Planned (M2/M3)
//!
//! * XYZ for molecular geometries and trajectories.
//! * NumPy-compatible array export used by the Python bridge.
//! * A container format for large volumetric datasets (HDF5 or equivalent).

use serde::{Deserialize, Serialize};

/// Reproducibility metadata attached to every exported calculation.
///
/// This type exists so that the framework cannot silently export a scientific
/// result without recording how it was produced. Fields that do not apply to a
/// given calculation are `None` rather than fabricated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalculationMetadata {
    /// Name of the physical model or solver that produced the result.
    pub model: String,
    /// Unit system in which the exported numbers are expressed.
    pub units: String,
    /// Software version string of the producing crate.
    pub software_version: String,
    /// Numerical parameters that affect the result (grid, tolerances, ...).
    pub numerical_parameters: Vec<(String, String)>,
    /// Basis set description, when the model uses one.
    pub basis_set: Option<String>,
    /// Convergence criterion description, when the model is iterative.
    pub convergence: Option<String>,
    /// Random seed, when the result depends on random sampling.
    pub random_seed: Option<u64>,
    /// Execution backend, e.g. `"cpu"`, `"wgpu/metal"`, `"cuda"`.
    pub backend: Option<String>,
    /// Free-form remarks about known limitations of the result.
    pub notes: Option<String>,
}

impl CalculationMetadata {
    /// Construct metadata with the required fields and no optional fields set.
    pub fn new(
        model: impl Into<String>,
        units: impl Into<String>,
        software_version: impl Into<String>,
    ) -> Self {
        Self {
            model: model.into(),
            units: units.into(),
            software_version: software_version.into(),
            numerical_parameters: Vec::new(),
            basis_set: None,
            convergence: None,
            random_seed: None,
            backend: None,
            notes: None,
        }
    }

    /// Add a numerical parameter record.
    pub fn with_parameter(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.numerical_parameters.push((key.into(), value.into()));
        self
    }

    /// Record the random seed used.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Record the execution backend.
    pub fn with_backend(mut self, backend: impl Into<String>) -> Self {
        self.backend = Some(backend.into());
        self
    }

    /// Record a limitation note.
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }

    /// Serialise to pretty-printed JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Parse from JSON.
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_round_trips_through_json() {
        let md =
            CalculationMetadata::new("hydrogenic-1e", "bohr/hartree", env!("CARGO_PKG_VERSION"))
                .with_parameter("grid", "128x128x128")
                .with_seed(20261015)
                .with_backend("cpu")
                .with_notes("unscreened one-electron model");
        let json = md.to_json().expect("serialise");
        let back = CalculationMetadata::from_json(&json).expect("deserialise");
        assert_eq!(md, back);
    }

    #[test]
    fn unset_optional_fields_are_null_not_fabricated() {
        let md = CalculationMetadata::new("model", "au", "0.1.0");
        let json = md.to_json().unwrap();
        assert!(json.contains("\"basis_set\": null"));
        assert!(json.contains("\"random_seed\": null"));
        assert!(json.contains("\"backend\": null"));
    }

    #[test]
    fn metadata_records_limitations_explicitly() {
        let md = CalculationMetadata::new("hydrogenic-1e", "au", "0.1.0")
            .with_notes("no screening, no electron correlation");
        assert_eq!(
            md.notes.as_deref(),
            Some("no screening, no electron correlation")
        );
    }
}
