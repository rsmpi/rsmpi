//! Bridge between rust types and raw values

use ffi::*;

use std;
use std::mem;
use std::os::raw::c_int;

/// Rust C bridge traits
pub mod traits {
    pub use super::{AsRaw, AsRawMut, Nullable};
}

/// A rust type than can identify as a raw value understood by the MPI C API.
pub unsafe trait AsRaw {
    /// The raw MPI C API type
    type Raw;
    /// The raw value
    fn as_raw(&self) -> Self::Raw;
}

unsafe impl<'a, T> AsRaw for &'a T where T: 'a + AsRaw
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

/// Applied to ffi handles that have a null value.
pub trait Nullable: Eq + Copy {
    /// Returns the null() value of this type.
    fn null() -> Self;

    /// Returns true if this value is null. Returns false if it is not null.
    fn is_null(&self) -> bool {
        *self == Self::null()
    }
}

impl Nullable for MPI_Request {
    fn null() -> Self {
        unsafe_extern_static!(RSMPI_REQUEST_NULL)
    }
}

/// Test to see if the request has been completed. If it has been, request
/// is set to RSMPI_REQUEST_NULL, and Some(status) is returned. If not, request
/// is not modified, and None is returned.
pub fn test(request: &mut MPI_Request) -> Option<MPI_Status> {
    unsafe {
        let mut status: MPI_Status = mem::uninitialized();
        let mut flag: c_int = mem::uninitialized();
        MPI_Test(request, &mut flag, &mut status);
        if flag != 0 {
            // persistent requests are not supported
            assert!(request.is_null());
            Some(status)
        } else {
            None
        }
    }
}

/// Wait for the request to finish and unregister the request object from its scope.
/// If provided, the status is written to the referent of the given reference.
/// The referent `MPI_Status` object is never read.
/// 
/// Prefer `Request::wait` or `Request::wait_without_status`.
pub fn wait_with(request: &mut MPI_Request, status: Option<&mut MPI_Status>) {
    unsafe {
        let status = match status {
            Some(r) => r,
            None => RSMPI_STATUS_IGNORE,
        };
        MPI_Wait(request, status);
        debug_assert!(request.is_null());  // persistent requests are not supported
    }
}

/// Thin wrapper for MPI_Waitall. Uses the native handle types directly.
/// 
/// Prefer `RequestCollection::wait_all_with_status`.
pub fn wait_all_with(requests: &mut [MPI_Request], statuses: Option<&mut [MPI_Status]>) {
    assert!(
        requests.len() <= std::i32::MAX as usize,
        "MPI can only index arrays up to size i32::MAX");

    let statuses_ptr = match statuses {
        Some(statuses) => {
            assert!(statuses.len() >= requests.len());
            statuses.as_mut_ptr()
        },
        None => unsafe_extern_static!(RSMPI_STATUSES_IGNORE),
    };

    unsafe {
        MPI_Waitall(requests.len() as i32, requests.as_mut_ptr(), statuses_ptr);
    }

    debug_assert!(requests.iter().all(|r| r.is_null()));
}