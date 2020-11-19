//! Internal module implementing a checking "boolean" type that can be used safely in MPI
//! communciation.
//!
//! `Bool` is exported directly from root of module.

use std::{error::Error, fmt::Display};

use crate::{
    datatype::SystemDatatype,
    traits::{Equivalence, EquivalenceFromAnyBytes},
};

/// This type provides safe boolean sends and receives in MPI by checking at runtime that only
/// valid bit representations (0 and 1) are used.
#[derive(Copy, Clone, Debug, Default)]
#[repr(transparent)]
pub struct Bool(u8);

/// Error related to invalid `Bool` objects.
#[allow(missing_copy_implementations)]
#[derive(Debug)]
pub struct BoolError(u8);

impl BoolError {
    /// Get the invalid value held by `Bool`.
    pub fn invalid_value(&self) -> u8 {
        self.0
    }
}

impl Display for BoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mpi::Bool contained illegal value 0x{:x}", self.0)
    }
}

impl Error for BoolError {}

impl Bool {
    /// Constructs a `Bool` object
    pub fn new(value: bool) -> Self {
        Self(value as u8)
    }

    /// If this `Bool` is valid, returns its actual value. Otherwise returns `None`.
    pub fn valid(&self) -> Result<bool, BoolError> {
        match self {
            Bool(0) => Ok(false),
            Bool(1) => Ok(true),
            Bool(x) => Err(BoolError(*x)),
        }
    }
}

unsafe impl Equivalence for Bool {
    type Out = SystemDatatype;

    fn equivalent_datatype() -> Self::Out {
        bool::equivalent_datatype()
    }
}

unsafe impl EquivalenceFromAnyBytes for Bool {}

impl Display for Bool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Ok(b) = self.valid() {
            b.fmt(f)
        } else {
            Display::fmt("invalid", f)
        }
    }
}

/// `Bool` has only partial equivalence similar to float. Invalid value comparisons are always
/// inequivalent.
impl PartialEq<Bool> for Bool {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Bool(0), Bool(0)) => true,
            (Bool(1), Bool(1)) => true,
            _ => false,
        }
    }
}

/// `Bool` has only partial equivalence similar to float. Invalid value comparisons are always
/// disordered.
impl PartialOrd<Bool> for Bool {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.0 > 1 || other.0 > 1 {
            return None;
        }

        self.0.partial_cmp(&other.0)
    }
}

/// `Bool` has only partial equivalence similar to float. Invalid value comparisons are always
/// inequivalent.
impl PartialEq<bool> for Bool {
    fn eq(&self, other: &bool) -> bool {
        self.eq(&Bool::new(*other))
    }
}

/// `Bool` has only partial equivalence similar to float. Invalid value comparisons are always
/// disordered.
impl PartialOrd<bool> for Bool {
    fn partial_cmp(&self, other: &bool) -> Option<std::cmp::Ordering> {
        self.partial_cmp(&Bool::new(*other))
    }
}

/// `Bool` has only partial equivalence similar to float. Invalid value comparisons are always
/// inequivalent.
impl PartialEq<Bool> for bool {
    fn eq(&self, other: &Bool) -> bool {
        Bool::new(*self).eq(other)
    }
}

/// `Bool` has only partial equivalence similar to float. Invalid value comparisons are always
/// disordered.
impl PartialOrd<Bool> for bool {
    fn partial_cmp(&self, other: &Bool) -> Option<std::cmp::Ordering> {
        Bool::new(*self).partial_cmp(other)
    }
}

impl From<bool> for Bool {
    fn from(b: bool) -> Self {
        Bool::new(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid() {
        assert!(Bool::new(false).valid().is_ok());
        assert!(Bool::new(true).valid().is_ok());

        // It is safe to transmute `Bool` from a `u8`.
        let bad: Bool = unsafe { std::mem::transmute(0xFFu8) };
        assert!(bad.valid().is_err());

        let err = bad.valid().err().unwrap();
        assert_eq!(0xFFu8, err.invalid_value());
    }

    #[test]
    fn check_impl() {
        // PartialEq<Bool> for Bool
        assert_eq!(Bool::new(true), Bool::new(true));
        assert_eq!(Bool::new(false), Bool::new(false));
        assert_ne!(Bool::new(true), Bool::new(false));
        assert_ne!(Bool::new(false), Bool::new(true));

        // PartialEq<Bool> for bool
        assert_eq!(true, Bool::new(true));
        assert_eq!(false, Bool::new(false));

        // PartialEq<bool> for Bool
        assert_eq!(Bool::new(true), true);
        assert_eq!(Bool::new(false), false);

        // PartialOrd
        assert!(Bool::new(false) < true);
        assert!(Bool::new(false) < Bool::new(true));
        assert!(false < Bool::new(true));

        assert!(Bool::new(true) > false);
        assert!(Bool::new(true) > Bool::new(false));
        assert!(true > Bool::new(false));

        // It is safe to transmute `Bool` from a `u8`.
        let bad1: Bool = unsafe { std::mem::transmute(0xFFu8) };
        let bad2: Bool = unsafe { std::mem::transmute(0x3u8) };

        assert_ne!(bad1, bad2);

        assert_ne!(true, bad1);
        assert_ne!(false, bad1);
        assert_ne!(Bool::new(true), bad1);
        assert_ne!(Bool::new(false), bad1);

        assert!(!(true < bad1));
        assert!(!(true <= bad1));
        assert!(!(true > bad1));
        assert!(!(true >= bad1));
        assert!(!(false < bad1));
        assert!(!(false <= bad1));
        assert!(!(false > bad1));
        assert!(!(false >= bad1));
    }

    #[test]
    fn unwrap_or() {
        assert!(Bool::new(true).valid().unwrap_or(false));
        assert!(!Bool::new(false).valid().unwrap_or(true));

        // It is safe to transmute `Bool` from a `u8`.
        let bad: Bool = unsafe { std::mem::transmute(0xFFu8) };

        assert!(bad.valid().unwrap_or(true));
        assert!(!bad.valid().unwrap_or(false));
    }

    #[test]
    #[should_panic(expected = "called `Result::unwrap()` on an `Err` value: BoolError(255)")]
    fn unwrap_invalid() {
        // It is safe to transmute `Bool` from a `u8`.
        let x: Bool = unsafe { std::mem::transmute(0xFFu8) };
        x.valid().unwrap();
    }
}
