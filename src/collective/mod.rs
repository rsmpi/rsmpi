//! Collective communication
//!
//! Developing...
//!
//! # Unfinished features
//!
//! - **5.5**: Varying counts gather operation, `MPI_Gatherv()`
//! - **5.6**: Scatter, `MPI_Scatterv()`
//! - **5.7**: Gather-to-all, `MPI_Allgatherv()`
//! - **5.8**: All-to-all, `MPI_Alltoallv()`, `MPI_Alltoallw()`
//! - **5.9**: Global reduction operations, `MPI_Op_create()`, `MPI_Op_free()`,
//! `MPI_Op_commutative()`
//! - **5.10**: Reduce-scatter, `MPI_Reduce_scatter_block()`, `MPI_Reduce_scatter()`
//! - **5.12**: Nonblocking collective operations, `MPI_Igatherv()`, `MPI_Iscatterv()`,
//! `MPI_Iallgatherv()`, `MPI_Ialltoallv()`, `MPI_Ialltoallw()`,
//! `MPI_Ireduce()`, `MPI_Iallreduce()`, `MPI_Ireduce_scatter_block()`, `MPI_Ireduce_scatter()`,
//! `MPI_Iscan()`, `MPI_Iexscan()`

use std::{mem, ptr};
use std::marker::PhantomData;

use ffi;
use ffi::{MPI_Request, MPI_Op};

use datatype::traits::*;
use raw::traits::*;
use topology::traits::*;
use topology::{Rank, Identifier};

pub mod traits;

/// Barrier synchronization among all processes in a `Communicator`
///
/// Calling processes (or threads within the calling processes) will enter the barrier and block
/// execution until all processes in the `Communicator` `&self` have entered the barrier.
///
/// # Standard section(s)
///
/// 5.3
pub trait Barrier {
    /// Partake in a barrier synchronization across all processes in the `Communicator` `&self`.
    ///
    /// # Examples
    ///
    /// See `examples/barrier.rs`
    fn barrier(&self);
}

impl<C: Communicator> Barrier for C {
    fn barrier(&self) {
        unsafe { ffi::MPI_Barrier(self.communicator().as_raw()); }
    }
}

/// Something that can take the role of 'root' in a collective operation.
///
/// Many collective operations define a 'root' process that takes a special role in the
/// communication. These collective operations are implemented as traits that have blanket
/// implementations for every type that implements the `Root` trait.
pub trait Root: Communicator {
    /// Rank of the root process
    fn root_rank(&self) -> Rank;
}

impl<'a, C: 'a + RawCommunicator> Root for Identifier<'a, C> {
    fn root_rank(&self) -> Rank {
        self.rank()
    }
}

/// Broadcast of the contents of a buffer
///
/// After the call completes, the `Buffer` on all processes in the `Communicator` of the `Root`
/// `&self` will contain what it contains on the `Root`.
///
/// # Standard section(s)
///
/// 5.4
pub trait BroadcastInto {
    /// Broadcast the contents of `buffer` from the `Root` to the `buffer`s on all other processes.
    ///
    /// # Examples
    ///
    /// See `examples/broadcast.rs`
    fn broadcast_into<Buf: BufferMut + ?Sized>(&self, buffer: &mut Buf);
}

impl<T: Root> BroadcastInto for T {
    fn broadcast_into<Buf: BufferMut + ?Sized>(&self, buffer: &mut Buf) {
        unsafe {
            ffi::MPI_Bcast(buffer.pointer_mut(), buffer.count(), buffer.datatype().as_raw(),
                self.root_rank(), self.communicator().as_raw());
        }
    }
}

// TODO: Introduce "partitioned buffer" for varying count gather/scatter/alltoall?

/// Gather contents of buffers on `Root`.
///
/// After the call completes, the contents of the `Buffer`s on all ranks will be
/// concatenated into the `Buffer` on `Root`.
///
/// # Standard section(s)
///
/// 5.5
pub trait GatherInto {
    /// Gather the contents of all `sendbuf`s on `Root` `&self`.
    ///
    /// This function must be called on all non-root processes.
    ///
    /// # Examples
    ///
    /// See `examples/gather.rs`
    fn gather_into<S: Buffer + ?Sized>(&self, sendbuf: &S);

    /// Gather the contents of all `sendbuf`s into `recvbuf` on `Root` `&self`.
    ///
    /// This function must be called on the root process.
    ///
    /// # Examples
    ///
    /// See `examples/gather.rs`
    fn gather_into_root<S: Buffer + ?Sized, R: BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: &mut R);
}

impl<T: Root> GatherInto for T {
    fn gather_into<S: Buffer + ?Sized>(&self, sendbuf: &S) {
        assert!(self.communicator().rank() != self.root_rank());
        unsafe {
            ffi::MPI_Gather(sendbuf.pointer(), sendbuf.count(), sendbuf.datatype().as_raw(),
                ptr::null_mut(), 0, u8::equivalent_datatype().as_raw(),
                self.root_rank(), self.communicator().as_raw());
        }
    }

    fn gather_into_root<S: Buffer + ?Sized, R: BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: &mut R) {
        assert!(self.communicator().rank() == self.root_rank());
        unsafe {
            let recvcount = recvbuf.count() / self.communicator().size();
            ffi::MPI_Gather(sendbuf.pointer(), sendbuf.count(), sendbuf.datatype().as_raw(),
                recvbuf.pointer_mut(), recvcount, recvbuf.datatype().as_raw(),
                self.root_rank(), self.communicator().as_raw());
        }
    }
}

/// Gather contents of buffers on all participating processes.
///
/// After the call completes, the contents of the send `Buffer`s on all processes will be
/// concatenated into the receive `Buffer`s on all ranks.
///
/// # Standard section(s)
///
/// 5.7
pub trait AllGatherInto {
    /// Gather the contents of all `sendbuf`s into all `rcevbuf`s on all processes in the
    /// communicator.
    ///
    /// # Examples
    ///
    /// See `examples/all_gather.rs`
    fn all_gather_into<S: Buffer + ?Sized, R: BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: &mut R);
}

impl<C: Communicator> AllGatherInto for C {
    fn all_gather_into<S: Buffer + ?Sized, R: BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: &mut R) {
        unsafe {
            ffi::MPI_Allgather(sendbuf.pointer(), sendbuf.count(), sendbuf.datatype().as_raw(),
                recvbuf.pointer_mut(), recvbuf.count() / self.communicator().size(),
                recvbuf.datatype().as_raw(), self.communicator().as_raw());
        }
    }
}

/// Scatter contents of a buffer on the root process to all processes.
///
/// After the call completes each participating process will have received a part of the send
/// `Buffer` on the root process.
///
/// # Standard section(s)
///
/// 5.6
pub trait ScatterInto {
    /// Scatter data from the root process to all participating processes.
    ///
    /// This function must be called on all non-root processes.
    ///
    /// # Examples
    ///
    /// See `examples/scatter.rs`
    fn scatter_into<R: BufferMut + ?Sized>(&self, recvbuf: &mut R);

    /// Scatter the contents of `sendbuf` to all participating processes.
    ///
    /// This function must be called on the root process.
    ///
    /// # Examples
    ///
    /// See `examples/scatter.rs`
    fn scatter_into_root<S: Buffer + ?Sized, R: BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: &mut R);
}

impl<T: Root> ScatterInto for T {
    fn scatter_into<R: BufferMut + ?Sized>(&self, recvbuf: &mut R) {
        assert!(self.communicator().rank() != self.root_rank());
        unsafe {
            ffi::MPI_Scatter(ptr::null(), 0, u8::equivalent_datatype().as_raw(),
                recvbuf.pointer_mut(), recvbuf.count(), recvbuf.datatype().as_raw(),
                self.root_rank(), self.communicator().as_raw());
        }
    }

    fn scatter_into_root<S: Buffer + ?Sized, R: BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: &mut R) {
        assert!(self.communicator().rank() == self.root_rank());
        let sendcount = sendbuf.count() / self.communicator().size();
        unsafe {
            ffi::MPI_Scatter(sendbuf.pointer(), sendcount, sendbuf.datatype().as_raw(),
                recvbuf.pointer_mut(), recvbuf.count(), recvbuf.datatype().as_raw(),
                self.root_rank(), self.communicator().as_raw());
        }
    }
}

/// Distribute the send `Buffer`s from all processes to the receive `Buffer`s on all processes.
///
/// # Standard section(s)
///
/// 5.8
pub trait AllToAllInto {
    /// Distribute the `sendbuf` from all ranks to the `recvbuf` on all ranks.
    ///
    /// # Examples
    ///
    /// See `examples/all_to_all.rs`
    fn all_to_all_into<S: Buffer + ?Sized, R: BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: &mut R);
}

impl<C: Communicator> AllToAllInto for C {
    fn all_to_all_into<S: Buffer + ?Sized, R: BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: &mut R) {
        let c_size = self.communicator().size();
        unsafe {
            ffi::MPI_Alltoall(sendbuf.pointer(), sendbuf.count() / c_size, sendbuf.datatype().as_raw(),
                recvbuf.pointer_mut(), recvbuf.count() / c_size, recvbuf.datatype().as_raw(),
                self.communicator().as_raw());
        }
    }
}

/// A built-in operation like `MPI_SUM`
///
/// # Standard section(s)
///
/// 5.9.2
#[derive(Copy, Clone)]
pub struct SystemOperation(MPI_Op);

macro_rules! system_operation_constructors {
    ($($ctor:ident => $val:path),*) => (
        $(pub fn $ctor() -> SystemOperation {
            //! A built-in operation
            SystemOperation($val)
        })*
    )
}

impl SystemOperation {
    system_operation_constructors! {
        max => ffi::RSMPI_MAX,
        min => ffi::RSMPI_MIN,
        sum => ffi::RSMPI_SUM,
        product => ffi::RSMPI_PROD,
        logical_and => ffi::RSMPI_LAND,
        bitwise_and => ffi::RSMPI_BAND,
        logical_or => ffi::RSMPI_LOR,
        bitwise_or => ffi::RSMPI_BOR,
        logical_xor => ffi::RSMPI_LXOR,
        bitwise_xor => ffi::RSMPI_BXOR
    }
}

impl AsRaw for SystemOperation {
    type Raw = MPI_Op;
    unsafe fn as_raw(&self) -> Self::Raw { self.0 }
}

macro_rules! reduce_into_specializations {
    ($($name:ident => $operation:expr),*) => (
        $(fn $name<S: Buffer + ?Sized, R: BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: Option<&mut R>) {
            self.reduce_into(sendbuf, recvbuf, $operation)
        })*
    )
}

/// Perform a global reduction, storing the result on the `Root` process.
///
/// # Standard section(s)
///
/// 5.9.1
pub trait ReduceInto {
    /// Performs a global reduction under the operation `op` of the input data in `sendbuf` and
    /// stores the result on the `Root` process.
    ///
    /// This function must be called on all non-root processes.
    ///
    /// # Examples
    ///
    /// See `examples/reduce.rs`
    fn reduce_into<S: Buffer + ?Sized, O: RawOperation>(&self, sendbuf: &S, op: O);

    /// Performs a global reduction under the operation `op` of the input data in `sendbuf` and
    /// stores the result in `recvbuf` on the `Root` process.
    ///
    /// This function must be called on the root process.
    ///
    /// # Examples
    ///
    /// See `examples/reduce.rs`
    fn reduce_into_root<S: Buffer + ?Sized, R: BufferMut + ?Sized, O: RawOperation>(&self, sendbuf: &S, recvbuf: &mut R, op: O);

//    reduce_into_specializations! {
//        max_into => SystemOperation::max(),
//        min_into => SystemOperation::min(),
//        sum_into => SystemOperation::sum(),
//        product_into => SystemOperation::product(),
//        logical_and_into => SystemOperation::logical_and(),
//        bitwise_and_into => SystemOperation::bitwise_and(),
//        logical_or_into => SystemOperation::logical_or(),
//        bitwise_or_into => SystemOperation::bitwise_or(),
//        logical_xor_into => SystemOperation::logical_xor(),
//        bitwise_xor_into => SystemOperation::bitwise_xor()
//    }
}

impl<T: Root> ReduceInto for T {
    fn reduce_into<S: Buffer + ?Sized, O: RawOperation>(&self, sendbuf: &S, op: O) {
        assert!(self.communicator().rank() != self.root_rank());
        unsafe {
            ffi::MPI_Reduce(sendbuf.pointer(), ptr::null_mut(), sendbuf.count(), sendbuf.datatype().as_raw(),
                op.as_raw(), self.root_rank(), self.communicator().as_raw());
        }
    }

    fn reduce_into_root<S: Buffer + ?Sized, R: BufferMut + ?Sized, O: RawOperation>(&self, sendbuf: &S, recvbuf: &mut R, op: O) {
        assert!(self.communicator().rank() == self.root_rank());
        unsafe {
            ffi::MPI_Reduce(sendbuf.pointer(), recvbuf.pointer_mut(), sendbuf.count(), sendbuf.datatype().as_raw(),
                op.as_raw(), self.root_rank(), self.communicator().as_raw());
        }
    }
}

/// Perform a global reduction, storing the result on all processes.
///
/// # Standard section(s)
///
/// 5.9.6
pub trait AllReduceInto {
    /// Performs a global reduction under the operation `op` of the input data in `sendbuf` and
    /// stores the result in `recvbuf` on all processes.
    ///
    /// # Examples
    ///
    /// See `examples/reduce.rs`
    fn all_reduce_into<S: Buffer + ?Sized, R: BufferMut + ?Sized, O: RawOperation>(&self, sendbuf: &S, recvbuf: &mut R, op: O);
}

impl<C: Communicator> AllReduceInto for C {
    fn all_reduce_into<S: Buffer + ?Sized, R: BufferMut + ?Sized, O: RawOperation>(&self, sendbuf: &S, recvbuf: &mut R, op: O) {
        unsafe {
            ffi::MPI_Allreduce(sendbuf.pointer(), recvbuf.pointer_mut(), sendbuf.count(),
                sendbuf.datatype().as_raw(), op.as_raw(), self.communicator().as_raw());
        }
    }
}

/// Perform a local reduction.
///
/// # Standard section(s)
///
/// 5.9.7
///
/// # Examples
///
/// See `examples/redure.rs`
pub fn reduce_local_into<S: Buffer + ?Sized, R: BufferMut + ?Sized, O: RawOperation>(inbuf: &S, inoutbuf: &mut R, op: O) {
    unsafe {
        ffi::MPI_Reduce_local(inbuf.pointer(), inoutbuf.pointer_mut(), inbuf.count(),
          inbuf.datatype().as_raw(), op.as_raw());
    }
}

/// Perform a global inclusive prefix reduction.
///
/// # Standard section(s)
///
/// 5.11.1
pub trait ScanInto {
    /// Performs a global inclusive prefix reduction of the data in `sendbuf` into `recvbuf` under
    /// operation `op`.
    ///
    /// # Examples
    ///
    /// See `examples/scan.rs`
    fn scan_into<S: Buffer + ?Sized, R: BufferMut + ?Sized, O: RawOperation>(&self, sendbuf: &S, recvbuf: &mut R, op: O);
}

impl<C: Communicator> ScanInto for C {
    fn scan_into<S: Buffer + ?Sized, R: BufferMut + ?Sized, O: RawOperation>(&self, sendbuf: &S, recvbuf: &mut R, op: O) {
        unsafe {
            ffi::MPI_Scan(sendbuf.pointer(), recvbuf.pointer_mut(), sendbuf.count(),
                sendbuf.datatype().as_raw(), op.as_raw(), self.communicator().as_raw());
        }
    }
}

/// Perform a global exclusive prefix reduction.
///
/// # Standard section(s)
///
/// 5.11.2
pub trait ExclusiveScanInto {
    /// Performs a global exclusive prefix reduction of the data in `sendbuf` into `recvbuf` under
    /// operation `op`.
    ///
    /// # Examples
    ///
    /// See `examples/scan.rs`
    fn exclusive_scan_into<S: Buffer + ?Sized, R: BufferMut + ?Sized, O: RawOperation>(&self, sendbuf: &S, recvbuf: &mut R, op: O);
}

impl<C: Communicator> ExclusiveScanInto for C {
    fn exclusive_scan_into<S: Buffer + ?Sized, R: BufferMut + ?Sized, O: RawOperation>(&self, sendbuf: &S, recvbuf: &mut R, op: O) {
        unsafe {
            ffi::MPI_Exscan(sendbuf.pointer(), recvbuf.pointer_mut(), sendbuf.count(),
                sendbuf.datatype().as_raw(), op.as_raw(), self.communicator().as_raw());
        }
    }
}

/// A request object for an immediate (non-blocking) barrier operation
///
/// # Examples
///
/// See `examples/immediate_barrier.rs`
///
/// # Standard section(s)
///
/// 3.7.1
#[must_use]
pub struct BarrierRequest(MPI_Request);

impl Drop for BarrierRequest {
    fn drop(&mut self) {
        unsafe {
            assert!(self.as_raw() == ffi::RSMPI_REQUEST_NULL,
                "asynchronous barrier request dropped without ascertaining completion.");
        }
    }
}

impl AsRaw for BarrierRequest {
    type Raw = MPI_Request;
    unsafe fn as_raw(&self) -> Self::Raw { self.0 }
}

impl AsRawMut for BarrierRequest {
    unsafe fn as_raw_mut(&mut self) -> *mut <Self as AsRaw>::Raw { &mut (self.0) }
}

/// Non-blocking barrier synchronization among all processes in a `Communicator`
///
/// Calling processes (or threads within the calling processes) enter the barrier. Completion
/// methods on the associated request object will block until all processes have entered.
///
/// # Standard section(s)
///
/// 5.12.1
pub trait ImmediateBarrier {
    /// Partake in a barrier synchronization across all processes in the `Communicator` `&self`.
    ///
    /// # Examples
    ///
    /// See `examples/immediate_barrier.rs`
    fn immediate_barrier(&self) -> BarrierRequest;
}

impl<C: Communicator> ImmediateBarrier for C {
    fn immediate_barrier(&self) -> BarrierRequest {
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Ibarrier(self.communicator().as_raw(), &mut request); }
        BarrierRequest(request)
    }
}

/// A request object for an immediate (non-blocking) broadcast operation
///
/// # Examples
///
/// See `examples/immediate_broadcast.rs`
///
/// # Standard section(s)
///
/// 3.7.1
#[must_use]
pub struct BroadcastRequest<'b, Buf: 'b + BufferMut + ?Sized>(MPI_Request, PhantomData<&'b mut Buf>);

impl<'b, Buf: 'b + BufferMut + ?Sized> AsRaw for BroadcastRequest<'b, Buf> {
    type Raw = MPI_Request;
    unsafe fn as_raw(&self) -> Self::Raw { self.0 }
}

impl<'b, Buf: 'b + BufferMut + ?Sized> AsRawMut for BroadcastRequest<'b, Buf> {
    unsafe fn as_raw_mut(&mut self) -> *mut <Self as AsRaw>::Raw { &mut (self.0) }
}

impl<'b, Buf: 'b + BufferMut + ?Sized> Drop for BroadcastRequest<'b, Buf> {
    fn drop(&mut self) {
        unsafe {
            assert!(self.as_raw() == ffi::RSMPI_REQUEST_NULL,
                "asynchronous broadcast request dropped without ascertaining completion.");
        }
    }
}

/// Non-blocking broadcast of a value from a `Root` process to all other processes.
///
/// # Standard section(s)
///
/// 5.12.2
pub trait ImmediateBroadcastInto {
    /// Initiate broadcast of a value from the `Root` process to all other processes.
    ///
    /// # Examples
    ///
    /// See `examples/immediate_broadcast.rs`
    fn immediate_broadcast_into<'b, Buf: 'b + BufferMut + ?Sized>(&self, buf: &'b mut Buf) -> BroadcastRequest<'b, Buf>;
}

impl<R: Root> ImmediateBroadcastInto for R {
    fn immediate_broadcast_into<'b, Buf: 'b + BufferMut + ?Sized>(&self, buf: &'b mut Buf) -> BroadcastRequest<'b, Buf> {
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Ibcast(buf.pointer_mut(), buf.count(), buf.datatype().as_raw(),
                self.root_rank(), self.communicator().as_raw(), &mut request);
        }
        BroadcastRequest(request, PhantomData)
    }
}

/// A request object for an immediate (non-blocking) gather operation
///
/// # Examples
///
/// See `examples/immediate_gather.rs`
///
/// # Standard section(s)
///
/// 3.7.1
#[must_use]
pub struct GatherRequest<'s, S: 's + Buffer + ?Sized>(MPI_Request, PhantomData<&'s S>);

impl<'s, S: 's + Buffer + ?Sized> AsRaw for GatherRequest<'s, S> {
    type Raw = MPI_Request;
    unsafe fn as_raw(&self) -> Self::Raw { self.0 }
}

impl<'s, S: 's + Buffer + ?Sized> AsRawMut for GatherRequest<'s, S> {
    unsafe fn as_raw_mut(&mut self) -> *mut <Self as AsRaw>::Raw { &mut (self.0) }
}

impl<'s, S: 's + Buffer + ?Sized> Drop for GatherRequest<'s, S> {
    fn drop(&mut self) {
        unsafe {
            assert!(self.as_raw() == ffi::RSMPI_REQUEST_NULL,
                "asynchronous gather request dropped without ascertaining completion.");
        }
    }
}

/// A request object for an immediate (non-blocking) gather operation on the root process
///
/// # Examples
///
/// See `examples/immediate_gather.rs`
///
/// # Standard section(s)
///
/// 3.7.1
#[must_use]
pub struct GatherRootRequest<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized>(MPI_Request, PhantomData<&'s S>, PhantomData<&'r mut R>);

impl<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized> AsRaw for GatherRootRequest<'s, 'r, S, R> {
    type Raw = MPI_Request;
    unsafe fn as_raw(&self) -> Self::Raw { self.0 }
}

impl<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized> AsRawMut for GatherRootRequest<'s, 'r, S, R> {
    unsafe fn as_raw_mut(&mut self) -> *mut <Self as AsRaw>::Raw { &mut (self.0) }
}

impl<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized> Drop for GatherRootRequest<'s, 'r, S, R> {
    fn drop(&mut self) {
        unsafe {
            assert!(self.as_raw() == ffi::RSMPI_REQUEST_NULL,
                "asynchronous gather request dropped without ascertaining completion.");
        }
    }
}

/// Non-blocking gather of values at the `Root` process
///
/// # Standard section(s)
///
/// 5.12.3
pub trait ImmediateGatherInto {
    /// Initiate non-blocking gather of the contents of all `sendbuf`s on `Root` `&self`.
    ///
    /// This function must be called on all non-root processes.
    ///
    /// # Examples
    ///
    /// See `examples/immediate_gather.rs`
    fn immediate_gather_into<'s, S: 's + Buffer + ?Sized>(&self, sendbuf: &S) -> GatherRequest<'s, S>;

    /// Initiate non-blocking gather of the contents of all `sendbuf`s on `Root` `&self`.
    ///
    /// This function must be called on the root processes.
    ///
    /// # Examples
    ///
    /// See `examples/immediate_gather.rs`
    fn immediate_gather_into_root<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: &mut R) -> GatherRootRequest<'s, 'r, S, R>;
}

impl<T: Root> ImmediateGatherInto for T {
    fn immediate_gather_into<'s, S: 's + Buffer + ?Sized>(&self, sendbuf: &S) -> GatherRequest<'s, S> {
        assert!(self.communicator().rank() != self.root_rank());
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Igather(sendbuf.pointer(), sendbuf.count(), sendbuf.datatype().as_raw(),
                ptr::null_mut(), 0, u8::equivalent_datatype().as_raw(),
                self.root_rank(), self.communicator().as_raw(), &mut request);
        }
        GatherRequest(request, PhantomData)
    }

    fn immediate_gather_into_root<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: &mut R) -> GatherRootRequest<'s, 'r, S, R> {
        assert!(self.communicator().rank() == self.root_rank());
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            let recvcount = recvbuf.count() / self.communicator().size();
            ffi::MPI_Igather(sendbuf.pointer(), sendbuf.count(), sendbuf.datatype().as_raw(),
                recvbuf.pointer_mut(), recvcount, recvbuf.datatype().as_raw(),
                self.root_rank(), self.communicator().as_raw(), &mut request);
        }
        GatherRootRequest(request, PhantomData, PhantomData)
    }
}

/// A request object for an immediate (non-blocking) all-gather operation
///
/// # Examples
///
/// See `examples/immediate_all_gather.rs`
///
/// # Standard section(s)
///
/// 3.7.1
#[must_use]
pub struct AllGatherRequest<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized>(MPI_Request, PhantomData<&'s S>, PhantomData<&'r mut R>);

impl<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized> AsRaw for AllGatherRequest<'s, 'r, S, R> {
    type Raw = MPI_Request;
    unsafe fn as_raw(&self) -> Self::Raw { self.0 }
}

impl<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized> AsRawMut for AllGatherRequest<'s, 'r, S, R> {
    unsafe fn as_raw_mut(&mut self) -> *mut <Self as AsRaw>::Raw { &mut (self.0) }
}

impl<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized> Drop for AllGatherRequest<'s, 'r, S, R> {
    fn drop(&mut self) {
        unsafe {
            assert!(self.as_raw() == ffi::RSMPI_REQUEST_NULL,
                "asynchronous all-gather request dropped without ascertaining completion.");
        }
    }
}

/// Non-blocking gather of contents of buffers on all participating processes.
///
/// # Standard section(s)
///
/// 5.12.5
pub trait ImmediateAllGatherInto {
    /// Initiate non-blocking gather of the contents of all `sendbuf`s into all `rcevbuf`s on all
    /// processes in the communicator.
    ///
    /// # Examples
    ///
    /// See `examples/immediate_all_gather.rs`
    fn immediate_all_gather_into<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: &mut R) -> AllGatherRequest<'s, 'r, S, R>;
}

impl<C: Communicator> ImmediateAllGatherInto for C {
    fn immediate_all_gather_into<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: &mut R) -> AllGatherRequest<'s, 'r, S, R> {
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            let recvcount = recvbuf.count() / self.communicator().size();
            ffi::MPI_Iallgather(sendbuf.pointer(), sendbuf.count(), sendbuf.datatype().as_raw(),
                recvbuf.pointer_mut(), recvcount, recvbuf.datatype().as_raw(),
                self.communicator().as_raw(), &mut request);
        }
        AllGatherRequest(request, PhantomData, PhantomData)
    }
}

/// A request object for an immediate (non-blocking) scatter operation
///
/// # Examples
///
/// See `examples/immediate_scatter.rs`
///
/// # Standard section(s)
///
/// 3.7.1
#[must_use]
pub struct ScatterRequest<'r, R: 'r + BufferMut + ?Sized>(MPI_Request, PhantomData<&'r mut R>);

impl<'r, R: 'r + BufferMut + ?Sized> AsRaw for ScatterRequest<'r, R> {
    type Raw = MPI_Request;
    unsafe fn as_raw(&self) -> Self::Raw { self.0 }
}

impl<'r, R: 'r + BufferMut + ?Sized> AsRawMut for ScatterRequest<'r, R> {
    unsafe fn as_raw_mut(&mut self) -> *mut <Self as AsRaw>::Raw { &mut (self.0) }
}

impl<'r, R: 'r + BufferMut + ?Sized> Drop for ScatterRequest<'r, R> {
    fn drop(&mut self) {
        unsafe {
            assert!(self.as_raw() == ffi::RSMPI_REQUEST_NULL,
                "asynchronous scatter request dropped without ascertaining completion.");
        }
    }
}

/// A request object for an immediate (non-blocking) scatter operation on the root process
///
/// # Examples
///
/// See `examples/immediate_scatter.rs`
///
/// # Standard section(s)
///
/// 3.7.1
#[must_use]
pub struct ScatterRootRequest<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized>(MPI_Request, PhantomData<&'s S>, PhantomData<&'r mut R>);

impl<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized> AsRaw for ScatterRootRequest<'s, 'r, S, R> {
    type Raw = MPI_Request;
    unsafe fn as_raw(&self) -> Self::Raw { self.0 }
}

impl<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized> AsRawMut for ScatterRootRequest<'s, 'r, S, R> {
    unsafe fn as_raw_mut(&mut self) -> *mut <Self as AsRaw>::Raw { &mut (self.0) }
}

impl<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized> Drop for ScatterRootRequest<'s, 'r, S, R> {
    fn drop(&mut self) {
        unsafe {
            assert!(self.as_raw() == ffi::RSMPI_REQUEST_NULL,
                "asynchronous scatter request dropped without ascertaining completion.");
        }
    }
}

/// Non-blocking scatter of values from the `Root` process
///
/// # Standard section(s)
///
/// 5.12.4
pub trait ImmediateScatterInto {
    /// Initiate non-blocking scatter of the contents of `sendbuf` from `Root` `&self`.
    ///
    /// This function must be called on all non-root processes.
    ///
    /// # Examples
    ///
    /// See `examples/immediate_scatter.rs`
    fn immediate_scatter_into<'r, R: 'r + BufferMut + ?Sized>(&self, recvbuf: &mut R) -> ScatterRequest<'r, R>;

    /// Initiate non-blocking scatter of the contents of `sendbuf` from `Root` `&self`.
    ///
    /// This function must be called on the root processes.
    ///
    /// # Examples
    ///
    /// See `examples/immediate_scatter.rs`
    fn immediate_scatter_into_root<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: &mut R) -> ScatterRootRequest<'s, 'r, S, R>;
}

impl<T: Root> ImmediateScatterInto for T {
    fn immediate_scatter_into<'r, R: 'r + BufferMut + ?Sized>(&self, recvbuf: &mut R) -> ScatterRequest<'r, R> {
        assert!(self.communicator().rank() != self.root_rank());
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Iscatter(ptr::null(), 0, u8::equivalent_datatype().as_raw(),
                recvbuf.pointer_mut(), recvbuf.count(), recvbuf.datatype().as_raw(),
                self.root_rank(), self.communicator().as_raw(), &mut request);
        }
        ScatterRequest(request, PhantomData)
    }

    fn immediate_scatter_into_root<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: &mut R) -> ScatterRootRequest<'s, 'r, S, R> {
        assert!(self.communicator().rank() == self.root_rank());
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            let sendcount = sendbuf.count() / self.communicator().size();
            ffi::MPI_Iscatter(sendbuf.pointer(), sendcount, sendbuf.datatype().as_raw(),
                recvbuf.pointer_mut(), recvbuf.count(), recvbuf.datatype().as_raw(),
                self.root_rank(), self.communicator().as_raw(), &mut request);
        }
        ScatterRootRequest(request, PhantomData, PhantomData)
    }
}

/// A request object for an immediate (non-blocking) all-to-all operation
///
/// # Examples
///
/// See `examples/immediate_all_to_all.rs`
///
/// # Standard section(s)
///
/// 3.7.1
#[must_use]
pub struct AllToAllRequest<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized>(MPI_Request, PhantomData<&'s S>, PhantomData<&'r mut R>);

impl<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized> AsRaw for AllToAllRequest<'s, 'r, S, R> {
    type Raw = MPI_Request;
    unsafe fn as_raw(&self) -> Self::Raw { self.0 }
}

impl<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized> AsRawMut for AllToAllRequest<'s, 'r, S, R> {
    unsafe fn as_raw_mut(&mut self) -> *mut <Self as AsRaw>::Raw { &mut (self.0) }
}

impl<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized> Drop for AllToAllRequest<'s, 'r, S, R> {
    fn drop(&mut self) {
        unsafe {
            assert!(self.as_raw() == ffi::RSMPI_REQUEST_NULL,
                "asynchronous all-to-all request dropped without ascertaining completion.");
        }
    }
}

/// Non-blocking all-to-all communication.
///
/// # Standard section(s)
///
/// 5.12.6
pub trait ImmediateAllToAllInto {
    /// Initiate non-blocking all-to-all communication.
    ///
    /// # Examples
    ///
    /// See `examples/immediate_all_to_all.rs`
    fn immediate_all_to_all_into<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: &mut R) -> AllToAllRequest<'s, 'r, S, R>;
}

impl<C: Communicator> ImmediateAllToAllInto for C {
    fn immediate_all_to_all_into<'s, 'r, S: 's + Buffer + ?Sized, R: 'r + BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: &mut R) -> AllToAllRequest<'s, 'r, S, R> {
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        let c_size = self.communicator().size();
        unsafe {
            ffi::MPI_Ialltoall(sendbuf.pointer(), sendbuf.count() / c_size, sendbuf.datatype().as_raw(),
                recvbuf.pointer_mut(), recvbuf.count() / c_size, recvbuf.datatype().as_raw(),
                self.communicator().as_raw(), &mut request);
        }
        AllToAllRequest(request, PhantomData, PhantomData)
    }
}
