//! Bridge between rust types and raw values

/// Rust C bridge traits
pub mod traits {
    pub use super::{AsRaw, AsRawMut};
}

/// A rust type than can identify as a raw value understood by the MPI C API.
pub unsafe trait AsRaw {
    /// The raw MPI C API type
    type Raw;
    /// The raw value
    fn as_raw(&self) -> Self::Raw;
}

unsafe impl<'a, T> AsRaw for &'a T
where
    T: 'a + AsRaw,
{
    type Raw = <T as AsRaw>::Raw;
    fn as_raw(&self) -> Self::Raw {
        (*self).as_raw()
    }
}

/// A rust type than can provide a mutable pointer to a raw value understood by the MPI C API.
pub unsafe trait AsRawMut: AsRaw {
    /// A mutable pointer to the raw value
    fn as_raw_mut(&mut self) -> *mut <Self as AsRaw>::Raw;
}
