//! Renderer errors.
//!
//! Every failure mode a renderer can encounter is named explicitly, including
//! the ones that only occur on machines without a usable GPU. Nothing in this
//! crate panics because a graphics capability is missing.

use std::fmt;

/// Everything that can go wrong while preparing or executing a render.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderError {
    /// No adapter satisfied the requested options.
    ///
    /// This is the expected outcome on a machine with no GPU, or when a fallback
    /// adapter was required and none exists. It is **not** a bug.
    NoAdapter {
        /// Whether a software/fallback adapter was required.
        required_fallback: bool,
    },
    /// An adapter was found but the device could not be created.
    DeviceCreationFailed {
        /// The message reported by the backend.
        message: String,
    },
    /// The adapter does not offer something that was explicitly required.
    UnsupportedCapability {
        /// What was required.
        what: String,
        /// What the adapter offers instead, where meaningful.
        available: String,
    },
    /// A resource would exceed a device limit.
    ResourceLimit {
        /// Name of the limit.
        limit: String,
        /// Value requested.
        requested: u64,
        /// Value supported by the device.
        available: u64,
    },
    /// A volume buffer was not a valid size for the grid it describes.
    VolumeLength {
        /// Expected number of values.
        expected: usize,
        /// Number of values supplied.
        got: usize,
    },
    /// A volume value was not finite.
    NonFiniteValue {
        /// Linear index (public order) of the offending value.
        index: usize,
    },
    /// A negative value was supplied where a probability density was required.
    NegativeValue {
        /// Linear index (public order) of the offending value.
        index: usize,
    },
    /// Grid geometry was degenerate or not finite.
    InvalidGeometry {
        /// Which field was rejected.
        field: &'static str,
    },
    /// A slice index or axis was outside the volume.
    OutOfRange {
        /// Description of the index.
        what: String,
        /// Requested value.
        requested: usize,
        /// Exclusive upper bound.
        limit: usize,
    },
    /// Reading back a rendered texture failed.
    ReadbackFailed {
        /// The message reported by the backend.
        message: String,
    },
    /// An internal invariant was violated. Indicates a bug in this crate.
    Internal {
        /// Description of the violated invariant.
        message: String,
    },
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RenderError::NoAdapter { required_fallback } => {
                if *required_fallback {
                    write!(
                        f,
                        "no fallback (software) GPU adapter is available on this system"
                    )
                } else {
                    write!(f, "no usable GPU adapter was found")
                }
            }
            RenderError::DeviceCreationFailed { message } => {
                write!(f, "the GPU device could not be created: {message}")
            }
            RenderError::UnsupportedCapability { what, available } => {
                write!(
                    f,
                    "the adapter does not support {what} (available: {available})"
                )
            }
            RenderError::ResourceLimit {
                limit,
                requested,
                available,
            } => write!(
                f,
                "resource limit exceeded: {limit} = {requested} requested, {available} available"
            ),
            RenderError::VolumeLength { expected, got } => write!(
                f,
                "volume has {expected} grid points but {got} values were supplied"
            ),
            RenderError::NonFiniteValue { index } => {
                write!(f, "volume value at index {index} is not finite")
            }
            RenderError::NegativeValue { index } => write!(
                f,
                "volume value at index {index} is negative; a probability density cannot be"
            ),
            RenderError::InvalidGeometry { field } => {
                write!(
                    f,
                    "grid geometry field `{field}` is degenerate or not finite"
                )
            }
            RenderError::OutOfRange {
                what,
                requested,
                limit,
            } => write!(f, "{what} = {requested} is outside [0, {limit})"),
            RenderError::ReadbackFailed { message } => {
                write!(f, "reading the rendered texture back failed: {message}")
            }
            RenderError::Internal { message } => write!(f, "internal error: {message}"),
        }
    }
}

impl std::error::Error for RenderError {}

impl RenderError {
    /// Whether the failure is simply "this machine has no usable GPU".
    ///
    /// Callers use this to decide whether to fall back to the CPU path without
    /// treating the situation as an error worth surfacing.
    pub fn is_unavailable(&self) -> bool {
        matches!(
            self,
            RenderError::NoAdapter { .. } | RenderError::DeviceCreationFailed { .. }
        )
    }

    /// Whether the caller asked for something the hardware cannot do, as opposed
    /// to the hardware being absent.
    pub fn is_capability_mismatch(&self) -> bool {
        matches!(
            self,
            RenderError::UnsupportedCapability { .. } | RenderError::ResourceLimit { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_errors_are_classified() {
        assert!(RenderError::NoAdapter {
            required_fallback: false
        }
        .is_unavailable());
        assert!(RenderError::DeviceCreationFailed {
            message: "x".into()
        }
        .is_unavailable());
        assert!(!RenderError::NoAdapter {
            required_fallback: false
        }
        .is_capability_mismatch());
    }

    #[test]
    fn capability_errors_are_classified() {
        assert!(RenderError::UnsupportedCapability {
            what: "Float32Filterable".into(),
            available: "not supported".into()
        }
        .is_capability_mismatch());
        assert!(RenderError::ResourceLimit {
            limit: "max_buffer_size".into(),
            requested: 10,
            available: 5
        }
        .is_capability_mismatch());
        assert!(!RenderError::ResourceLimit {
            limit: "l".into(),
            requested: 10,
            available: 5
        }
        .is_unavailable());
    }

    #[test]
    fn messages_name_the_problem() {
        let e = RenderError::OutOfRange {
            what: "slice index".into(),
            requested: 7,
            limit: 5,
        };
        let s = e.to_string();
        assert!(s.contains("7") && s.contains("5") && s.contains("slice index"));

        let e = RenderError::ResourceLimit {
            limit: "max_storage_buffer_binding_size".into(),
            requested: 200_000_000,
            available: 134_217_728,
        };
        let s = e.to_string();
        assert!(s.contains("200000000") && s.contains("134217728"));
    }

    #[test]
    fn errors_are_std_errors() {
        fn takes_std_error(_: &dyn std::error::Error) {}
        takes_std_error(&RenderError::Internal {
            message: "x".into(),
        });
    }
}
