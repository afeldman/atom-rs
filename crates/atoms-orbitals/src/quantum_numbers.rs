//! Quantum numbers and their constraints.
//!
//! For the hydrogen-like problem the quantum numbers satisfy
//!
//! * `n >= 1` — principal
//! * `0 <= l < n` — orbital angular momentum
//! * `-l <= m <= l` — magnetic
//!
//! [`QuantumNumbers`] is the single place where these constraints are enforced.
//! Every public entry point in this crate that needs quantum numbers takes a
//! validated `QuantumNumbers`, so an invalid combination cannot reach the
//! numerical kernels through the typed API.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Reasons a quantum-number triple can be rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantumNumberError {
    /// `n == 0`; the principal quantum number starts at 1.
    PrincipalMustBePositive,
    /// `l >= n`; the angular momentum is bounded by the principal number.
    AngularMomentumTooLarge {
        /// The requested `n`.
        n: u32,
        /// The requested `l`.
        l: u32,
    },
    /// `|m| > l`; the magnetic quantum number is bounded by `l`.
    MagneticOutOfRange {
        /// The requested `l`.
        l: u32,
        /// The requested `m`.
        m: i32,
    },
}

impl fmt::Display for QuantumNumberError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantumNumberError::PrincipalMustBePositive => {
                write!(
                    f,
                    "principal quantum number must satisfy n >= 1 (got n = 0)"
                )
            }
            QuantumNumberError::AngularMomentumTooLarge { n, l } => write!(
                f,
                "orbital angular momentum must satisfy 0 <= l < n (got l = {l}, n = {n})"
            ),
            QuantumNumberError::MagneticOutOfRange { l, m } => write!(
                f,
                "magnetic quantum number must satisfy -l <= m <= l (got m = {m}, l = {l})"
            ),
        }
    }
}

impl std::error::Error for QuantumNumberError {}

/// A validated triple `(n, l, m)`.
///
/// Construction is the only way to obtain this type, which guarantees the
/// invariants `n >= 1`, `l < n` and `|m| <= l` everywhere downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QuantumNumbers {
    n: u32,
    l: u32,
    m: i32,
}

impl QuantumNumbers {
    /// Validate and construct a quantum-number triple.
    ///
    /// # Errors
    /// Returns [`QuantumNumberError`] if `n == 0`, if `l >= n`, or if
    /// `|m| > l`.
    pub fn new(n: u32, l: u32, m: i32) -> Result<Self, QuantumNumberError> {
        if n == 0 {
            return Err(QuantumNumberError::PrincipalMustBePositive);
        }
        if l >= n {
            return Err(QuantumNumberError::AngularMomentumTooLarge { n, l });
        }
        if (m.unsigned_abs() as u64) > l as u64 {
            return Err(QuantumNumberError::MagneticOutOfRange { l, m });
        }
        Ok(Self { n, l, m })
    }

    /// Principal quantum number `n`.
    pub fn n(&self) -> u32 {
        self.n
    }

    /// Orbital angular momentum quantum number `l`.
    pub fn l(&self) -> u32 {
        self.l
    }

    /// Magnetic quantum number `m`, which may be negative or zero.
    pub fn m(&self) -> i32 {
        self.m
    }

    /// Number of radial nodes, `n - l - 1`.
    pub fn radial_nodes(&self) -> u32 {
        self.n - self.l - 1
    }

    /// Number of angular nodal surfaces (planes or cones), equal to `l`.
    pub fn angular_nodes(&self) -> u32 {
        self.l
    }

    /// Total number of nodes, `n - 1`.
    pub fn total_nodes(&self) -> u32 {
        self.n - 1
    }

    /// Spectroscopic letter for `l`: `s, p, d, f, g, h, i, k, ...`.
    ///
    /// Follows the conventional sequence and falls back to `"?"` for `l > 7`.
    pub fn subshell_letter(&self) -> char {
        match self.l {
            0 => 's',
            1 => 'p',
            2 => 'd',
            3 => 'f',
            4 => 'g',
            5 => 'h',
            6 => 'i',
            7 => 'k',
            _ => '?',
        }
    }

    /// Designation such as `"1s"`, `"2p"`, `"3d"`. Negative `m` is included as
    /// a suffix only when it is non-zero, e.g. `"2p m=-1"`.
    pub fn designation(&self) -> String {
        if self.m == 0 {
            format!("{}{}", self.n, self.subshell_letter())
        } else {
            format!("{}{} (m={})", self.n, self.subshell_letter(), self.m)
        }
    }

    /// All `(l, m)` pairs for a given `n`, in the order
    /// `l = 0..n-1`, `m = -l..l`. There are `n^2` of them.
    pub fn shell(n: u32) -> Vec<Self> {
        let mut out = Vec::new();
        if n == 0 {
            return out;
        }
        for l in 0..n {
            let lm = l as i32;
            for m in -lm..=lm {
                out.push(Self { n, l, m });
            }
        }
        out
    }
}

impl fmt::Display for QuantumNumbers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(n={}, l={}, m={})", self.n, self.l, self.m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_valid_combinations() {
        for n in 1..=6u32 {
            for l in 0..n {
                let lm = l as i32;
                for m in -lm..=lm {
                    let q = QuantumNumbers::new(n, l, m).expect("valid");
                    assert_eq!((q.n(), q.l(), q.m()), (n, l, m));
                }
            }
        }
    }

    #[test]
    fn rejects_n_equal_zero() {
        assert_eq!(
            QuantumNumbers::new(0, 0, 0),
            Err(QuantumNumberError::PrincipalMustBePositive)
        );
    }

    #[test]
    fn rejects_l_greater_or_equal_n() {
        assert_eq!(
            QuantumNumbers::new(1, 1, 0),
            Err(QuantumNumberError::AngularMomentumTooLarge { n: 1, l: 1 })
        );
        assert_eq!(
            QuantumNumbers::new(2, 5, 0),
            Err(QuantumNumberError::AngularMomentumTooLarge { n: 2, l: 5 })
        );
    }

    #[test]
    fn rejects_m_outside_l() {
        assert_eq!(
            QuantumNumbers::new(2, 1, 2),
            Err(QuantumNumberError::MagneticOutOfRange { l: 1, m: 2 })
        );
        assert_eq!(
            QuantumNumbers::new(3, 2, -3),
            Err(QuantumNumberError::MagneticOutOfRange { l: 2, m: -3 })
        );
    }

    /// The `|m| <= l` bound must not be broken by negative `m` wraparound, which
    /// is the classic bug when converting `m` to an unsigned type.
    #[test]
    fn negative_m_is_validated_by_magnitude() {
        assert!(QuantumNumbers::new(3, 2, -2).is_ok());
        assert!(QuantumNumbers::new(3, 2, -1).is_ok());
        assert!(QuantumNumbers::new(3, 2, -3).is_err());
        // i32::MIN must not panic or wrap
        assert!(QuantumNumbers::new(5, 4, i32::MIN).is_err());
        assert!(QuantumNumbers::new(1, 0, -1).is_err());
    }

    #[test]
    fn node_counts_are_consistent() {
        let q = QuantumNumbers::new(4, 2, 0).unwrap();
        assert_eq!(q.radial_nodes(), 1);
        assert_eq!(q.angular_nodes(), 2);
        assert_eq!(q.total_nodes(), 3);
        // numbers of radial and angular nodes always add to n - 1
        for n in 1..=8u32 {
            for l in 0..n {
                let q = QuantumNumbers::new(n, l, 0).unwrap();
                assert_eq!(q.radial_nodes() + q.angular_nodes(), q.total_nodes());
                assert_eq!(q.total_nodes(), n - 1);
            }
        }
    }

    #[test]
    fn designations_are_conventional() {
        assert_eq!(QuantumNumbers::new(1, 0, 0).unwrap().designation(), "1s");
        assert_eq!(QuantumNumbers::new(2, 1, 0).unwrap().designation(), "2p");
        assert_eq!(QuantumNumbers::new(3, 2, 0).unwrap().designation(), "3d");
        assert_eq!(QuantumNumbers::new(4, 3, 0).unwrap().designation(), "4f");
        assert_eq!(
            QuantumNumbers::new(2, 1, -1).unwrap().designation(),
            "2p (m=-1)"
        );
    }

    #[test]
    fn shell_enumeration_has_n_squared_states() {
        for n in 1..=6u32 {
            let s = QuantumNumbers::shell(n);
            assert_eq!(s.len(), (n * n) as usize, "n={n}");
            // all m values appear for each l
            for l in 0..n {
                let count = s.iter().filter(|q| q.l() == l).count();
                assert_eq!(count, (2 * l + 1) as usize, "n={n} l={l}");
            }
        }
        assert!(QuantumNumbers::shell(0).is_empty());
    }

    #[test]
    fn error_messages_are_specific() {
        let e = QuantumNumbers::new(2, 1, 5).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("m = 5"), "{s}");
        assert!(s.contains("l = 1"), "{s}");
    }
}
