//! # Unfinished features
//!
//! - **3.7**: Nonblocking mode:
//!   - Completion, `MPI_Waitany()`, `MPI_Waitall()`, `MPI_Waitsome()`,
//!   `MPI_Testany()`, `MPI_Testall()`, `MPI_Testsome()`, `MPI_Request_get_status()`
//! - **3.8**:
//!   - Cancellation, `MPI_Cancel()`, `MPI_Test_cancelled()`

use std::mem;
use std::marker::PhantomData;

use libc::c_int;

use ffi;
use ffi::{MPI_Request, MPI_Status};

use datatype::traits::*;
use point_to_point::{Status};
use raw::traits::*;

pub mod traits;

/// Wait for an operation to finish.
///
/// # Examples
///
/// See `examples/immediate.rs`
///
/// # Standard section(s)
///
/// 3.7.3
pub trait Wait: RawRequest + Sized {
    /// Will block execution of the calling thread until the associated operation has finished.
    fn wait(mut self) -> Status {
        let mut status: MPI_Status = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Wait(self.as_raw_mut(), &mut status);
        }
        Status::from_raw(status)
    }
}

impl<R: RawRequest + Sized> Wait for R { }

/// Test whether an operation has finished.
///
/// # Examples
///
/// See `examples/immediate.rs`
///
/// # Standard section(s)
///
/// 3.7.3
pub trait Test: RawRequest + Sized {
    /// If the operation has finished returns the `Status` otherwise returns the unfinished
    /// `Request`.
    fn test(mut self) -> Result<Status, Self> {
        let mut status: MPI_Status = unsafe { mem::uninitialized() };
        let mut flag: c_int = 0;
        unsafe {
            ffi::MPI_Test(self.as_raw_mut(), &mut flag, &mut status);
        }
        if flag != 0 {
            Result::Ok(Status::from_raw(status))
        } else {
            Result::Err(self)
        }
    }
}

impl<R: RawRequest + Sized> Test for R { }

/// A request object for an non-blocking operation that holds no references
///
/// # Examples
///
/// See `examples/immediate_barrier.rs`
///
/// # Standard section(s)
///
/// 3.7.1
#[must_use]
pub struct Request(MPI_Request);

impl Request {
    /// Construct a request object from the raw MPI type
    pub fn from_raw(request: MPI_Request) -> Request {
        Request(request)
    }
}

impl AsRaw for Request {
    type Raw = MPI_Request;
    unsafe fn as_raw(&self) -> Self::Raw { self.0 }
}

impl AsRawMut for Request {
    unsafe fn as_raw_mut(&mut self) -> *mut <Self as AsRaw>::Raw { &mut (self.0) }
}

impl RawRequest for Request { }

impl Drop for Request {
    fn drop(&mut self) {
        unsafe {
            assert!(self.as_raw() == ffi::RSMPI_REQUEST_NULL,
                "request dropped without ascertaining completion.");
        }
    }
}

/// A request object for a non-blocking operation that holds a reference to an immutable buffer
///
/// # Examples
///
/// See `examples/immediate.rs`
///
/// # Standard section(s)
///
/// 3.7.1
#[must_use]
pub struct ReadRequest<'b, Buf: 'b + Buffer + ?Sized>(MPI_Request, PhantomData<&'b Buf>);

impl<'b, Buf: 'b + Buffer + ?Sized> ReadRequest<'b, Buf> {
    /// Construct a request object from the raw MPI type
    pub fn from_raw(request: MPI_Request, _: &'b Buf) -> ReadRequest<'b, Buf> {
        ReadRequest(request, PhantomData)
    }
}

impl<'b, Buf: 'b + Buffer + ?Sized> AsRaw for ReadRequest<'b, Buf> {
    type Raw = MPI_Request;
    unsafe fn as_raw(&self) -> Self::Raw { self.0 }
}

impl<'b, Buf: 'b + Buffer + ?Sized> AsRawMut for ReadRequest<'b, Buf> {
    unsafe fn as_raw_mut(&mut self) -> *mut <Self as AsRaw>::Raw { &mut (self.0) }
}

impl<'b, Buf: 'b + Buffer + ?Sized> RawRequest for ReadRequest<'b, Buf> { }

impl<'b, Buf: 'b + Buffer + ?Sized> Drop for ReadRequest<'b, Buf> {
    fn drop(&mut self) {
        unsafe {
            assert!(self.as_raw() == ffi::RSMPI_REQUEST_NULL,
                "read request dropped without ascertaining completion.");
        }
    }
}

/// A request object for a non-blocking operation that holds a reference to a mutable buffer
///
/// # Examples
///
/// See `examples/immediate.rs`
///
/// # Standard section(s)
///
/// 3.7.1
#[must_use]
pub struct WriteRequest<'b, Buf: 'b + BufferMut + ?Sized>(MPI_Request, PhantomData<&'b mut Buf>);

impl<'b, Buf: 'b + BufferMut + ?Sized> WriteRequest<'b, Buf> {
    /// Construct a request object from the raw MPI type
    pub fn from_raw(request: MPI_Request, _: &'b Buf) -> WriteRequest<'b, Buf> {
        WriteRequest(request, PhantomData)
    }
}

impl<'b, Buf: 'b + BufferMut + ?Sized> AsRaw for WriteRequest<'b, Buf> {
    type Raw = MPI_Request;
    unsafe fn as_raw(&self) -> Self::Raw { self.0 }
}

impl<'b, Buf: 'b + BufferMut + ?Sized> AsRawMut for WriteRequest<'b, Buf> {
    unsafe fn as_raw_mut(&mut self) -> *mut <Self as AsRaw>::Raw { &mut (self.0) }
}

impl<'b, Buf: 'b + BufferMut + ?Sized> RawRequest for WriteRequest<'b, Buf> { }

impl<'b, Buf: 'b + BufferMut + ?Sized> Drop for WriteRequest<'b, Buf> {
    fn drop(&mut self) {
        unsafe {
            assert!(self.as_raw() == ffi::RSMPI_REQUEST_NULL,
                "write request dropped without ascertaining completion.");
        }
    }
}

/// A request object for a non-blocking operation that holds a reference to a mutable and an
/// immutable buffer
///
/// # Examples
///
/// See `examples/immediate_gather.rs`
///
/// # Standard section(s)
///
/// 3.7.1
#[must_use]
pub struct ReadWriteRequest<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized>(MPI_Request, PhantomData<&'s S>, PhantomData<&'r mut R>);

impl<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized> ReadWriteRequest<'s, 'r, S, R> {
    /// Construct a request object from the raw MPI type
    pub fn from_raw(request: MPI_Request, _: &'s S, _: &'r R) -> ReadWriteRequest<'s, 'r, S, R> {
        ReadWriteRequest(request, PhantomData, PhantomData)
    }
}

impl<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized> AsRaw for ReadWriteRequest<'s, 'r, S, R> {
    type Raw = MPI_Request;
    unsafe fn as_raw(&self) -> Self::Raw { self.0 }
}

impl<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized> AsRawMut for ReadWriteRequest<'s, 'r, S, R> {
    unsafe fn as_raw_mut(&mut self) -> *mut <Self as AsRaw>::Raw { &mut (self.0) }
}

impl<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized> RawRequest for ReadWriteRequest<'s, 'r, S, R> { }

impl<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized> Drop for ReadWriteRequest<'s, 'r, S, R> {
    fn drop(&mut self) {
        unsafe {
            assert!(self.as_raw() == ffi::RSMPI_REQUEST_NULL,
                "read-write request dropped without ascertaining completion.");
        }
    }
}
