//! This module defines traits for safely sending and receiving buffers using MPI. Because MPI
//! doesn't require any sort of type checking, it can be essentially used as a way to transmute
//! between different types. Therefore, we must only allow sending values that don't expose, for
//! example, uninitialized padding, or receiving into types that don't allow any valid bit pattern.
//!
//! At the time of this writing (12/11/2019), preliminary investigations are being made into
//! providing "safe transmute" capabilities in the core language.
//! See:
//! - Pre-RFC: https://internals.rust-lang.org/t/pre-rfc-v2-safe-transmute/11431
//! - RFC Project Group: https://github.com/rust-lang/rfcs/pull/2835

use std::num;

pub(crate) mod traits {
    pub use super::{FromAnyBytes, ToBytes};
}

/// Any type that can be safely converted to bytes, which requires that the type has no padding.
pub unsafe trait ToBytes {}

/// Any type that can be composed of an arbitrary, properly-aligned and sized slice of bytes. This
/// includes any type where all fields are themselves `FromAnyBytes`, as long as all possible bit
/// patterns of those fields are allowed in any combination.
pub unsafe trait FromAnyBytes {}

unsafe impl ToBytes for bool {}
unsafe impl ToBytes for char {}

unsafe impl ToBytes for i8 {}
unsafe impl ToBytes for i16 {}
unsafe impl ToBytes for i32 {}
unsafe impl ToBytes for i64 {}
unsafe impl ToBytes for i128 {}
unsafe impl ToBytes for isize {}

unsafe impl ToBytes for u8 {}
unsafe impl ToBytes for u16 {}
unsafe impl ToBytes for u32 {}
unsafe impl ToBytes for u64 {}
unsafe impl ToBytes for u128 {}
unsafe impl ToBytes for usize {}

unsafe impl ToBytes for f32 {}
unsafe impl ToBytes for f64 {}

unsafe impl ToBytes for Option<num::NonZeroI8> {}
unsafe impl ToBytes for Option<num::NonZeroI16> {}
unsafe impl ToBytes for Option<num::NonZeroI32> {}
unsafe impl ToBytes for Option<num::NonZeroI64> {}
unsafe impl ToBytes for Option<num::NonZeroI128> {}
unsafe impl ToBytes for Option<num::NonZeroIsize> {}

unsafe impl ToBytes for Option<num::NonZeroU8> {}
unsafe impl ToBytes for Option<num::NonZeroU16> {}
unsafe impl ToBytes for Option<num::NonZeroU32> {}
unsafe impl ToBytes for Option<num::NonZeroU64> {}
unsafe impl ToBytes for Option<num::NonZeroU128> {}
unsafe impl ToBytes for Option<num::NonZeroUsize> {}

unsafe impl ToBytes for num::NonZeroI8 {}
unsafe impl ToBytes for num::NonZeroI16 {}
unsafe impl ToBytes for num::NonZeroI32 {}
unsafe impl ToBytes for num::NonZeroI64 {}
unsafe impl ToBytes for num::NonZeroI128 {}
unsafe impl ToBytes for num::NonZeroIsize {}

unsafe impl ToBytes for num::NonZeroU8 {}
unsafe impl ToBytes for num::NonZeroU16 {}
unsafe impl ToBytes for num::NonZeroU32 {}
unsafe impl ToBytes for num::NonZeroU64 {}
unsafe impl ToBytes for num::NonZeroU128 {}
unsafe impl ToBytes for num::NonZeroUsize {}

unsafe impl FromAnyBytes for i8 {}
unsafe impl FromAnyBytes for i16 {}
unsafe impl FromAnyBytes for i32 {}
unsafe impl FromAnyBytes for i64 {}
unsafe impl FromAnyBytes for i128 {}
unsafe impl FromAnyBytes for isize {}

unsafe impl FromAnyBytes for u8 {}
unsafe impl FromAnyBytes for u16 {}
unsafe impl FromAnyBytes for u32 {}
unsafe impl FromAnyBytes for u64 {}
unsafe impl FromAnyBytes for u128 {}
unsafe impl FromAnyBytes for usize {}

unsafe impl FromAnyBytes for f32 {}
unsafe impl FromAnyBytes for f64 {}

unsafe impl FromAnyBytes for Option<num::NonZeroI8> {}
unsafe impl FromAnyBytes for Option<num::NonZeroI16> {}
unsafe impl FromAnyBytes for Option<num::NonZeroI32> {}
unsafe impl FromAnyBytes for Option<num::NonZeroI64> {}
unsafe impl FromAnyBytes for Option<num::NonZeroI128> {}
unsafe impl FromAnyBytes for Option<num::NonZeroIsize> {}

unsafe impl FromAnyBytes for Option<num::NonZeroU8> {}
unsafe impl FromAnyBytes for Option<num::NonZeroU16> {}
unsafe impl FromAnyBytes for Option<num::NonZeroU32> {}
unsafe impl FromAnyBytes for Option<num::NonZeroU64> {}
unsafe impl FromAnyBytes for Option<num::NonZeroU128> {}
unsafe impl FromAnyBytes for Option<num::NonZeroUsize> {}
