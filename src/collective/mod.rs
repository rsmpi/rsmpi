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
//! - **5.12**: Nonblocking collective operations, `MPI_Ibcast()`,
//! `MPI_Igather()`, `MPI_Igatherv()`, `MPI_Iscatter()`, `MPI_Iscatterv()`, `MPI_Iallgather()`,
//! `MPI_Iallgatherv()`, `MPI_Ialltoall()`, `MPI_Ialltoallv()`, `MPI_Ialltoallw()`,
//! `MPI_Ireduce()`, `MPI_Iallreduce()`, `MPI_Ireduce_scatter_block()`, `MPI_Ireduce_scatter()`,
//! `MPI_Iscan()`, `MPI_Iexscan()`

use std::{mem, ptr};

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
    /// Gather the contents of all `sendbuf`s into `recvbuf` on `Root` `&self`.
    ///
    /// # Examples
    ///
    /// See `examples/gather.rs`
    fn gather_into<S: Buffer + ?Sized, R: BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: Option<&mut R>);
}

impl<T: Root> GatherInto for T {
    fn gather_into<S: Buffer + ?Sized, R: BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: Option<&mut R>) {
        unsafe {
            let (recvptr, recvcount, recvtype) = recvbuf.map_or(
                (ptr::null_mut(), 0, u8::equivalent_datatype().as_raw()),
                |x| (x.pointer_mut(), x.count() / self.communicator().size(), x.datatype().as_raw()));

            ffi::MPI_Gather(sendbuf.pointer(), sendbuf.count(), sendbuf.datatype().as_raw(),
                recvptr, recvcount, recvtype, self.root_rank(), self.communicator().as_raw());
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
    /// Scatter the contents of `sendbuf` to the participating processes.
    ///
    /// # Examples
    ///
    /// See `examples/scatter.rs`
    fn scatter_into<S: Buffer + ?Sized, R: BufferMut + ?Sized>(&self, sendbuf: Option<&S>, recvbuf: &mut R);
}

impl<T: Root> ScatterInto for T {
    fn scatter_into<S: Buffer + ?Sized, R: BufferMut + ?Sized>(&self, sendbuf: Option<&S>, recvbuf: &mut R) {
        unsafe {
            let (sendptr, sendcount, sendtype) = sendbuf.map_or(
                (ptr::null(), 0, u8::equivalent_datatype().as_raw()),
                |x| (x.pointer(), x.count() / self.communicator().size(), x.datatype().as_raw()));

            ffi::MPI_Scatter(sendptr, sendcount, sendtype,
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

impl RawOperation for SystemOperation { }

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
    /// stores the result in `recvbuf` on the `Root` process.
    ///
    /// `recvbuf` is ignored and can be `None` on all processes but the `Root` process.
    ///
    /// # Examples
    ///
    /// See `examples/reduce.rs`
    fn reduce_into<S: Buffer + ?Sized, R: BufferMut + ?Sized, O: RawOperation>(&self, sendbuf: &S, recvbuf: Option<&mut R>, op: O);

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
    fn reduce_into<S: Buffer + ?Sized, R: BufferMut + ?Sized, O: RawOperation>(&self, sendbuf: &S, recvbuf: Option<&mut R>, op: O) {
        unsafe {
            let recvptr = recvbuf.map_or(ptr::null_mut(), |x| { x.pointer_mut() });
            ffi::MPI_Reduce(sendbuf.pointer(), recvptr, sendbuf.count(), sendbuf.datatype().as_raw(),
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

impl RawRequest for BarrierRequest { }

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
