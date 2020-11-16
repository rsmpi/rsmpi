//! This module defines traits for safely receiving buffers using MPI. Because MPI
//! doesn't require any sort of type checking, it can potentially be used as a way to transmute
//! between different types. Therefore, we must only allow receiving into types that don't allow any
//! valid bit pattern.

use std::mem::MaybeUninit;

use crate::datatype::Equivalence;

pub(crate) mod traits {
    pub use super::EquivalenceFromAnyBytes;
}

/// Any type whose `Equivalence` implementation maps only fields that can be composed of an
/// arbitrary permutation of bytes. This includes any type where all fields are themselves
/// `EquivalenceFromAnyBytes`, as long as all possible bit patterns of those fields are allowed in
/// any combination.
pub unsafe trait EquivalenceFromAnyBytes {}

unsafe impl<T> EquivalenceFromAnyBytes for MaybeUninit<T> where T: Equivalence {}
