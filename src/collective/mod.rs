//! Collective communication
//!
//! Developing...
//!
//! # Unfinished features
//!
//! - **5.5**: Varying counts gather operation, `MPI_Gatherv()`
//! - **5.6**: Scatter, `MPI_Scatter()`, `MPI_Scatterv()`
//! - **5.7**: Gather-to-all, `MPI_Allgatherv()`
//! - **5.8**: All-to-all, `MPI_Alltoall()`, `MPI_Alltoallv()`, `MPI_Alltoallw()`
//! - **5.9**: Global reduction operations, `MPI_Reduce()`, `MPI_Op_create()`, `MPI_Op_free()`,
//! `MPI_Allreduce()`, `MPI_Reduce_local()`, `MPI_Op_commutative()`
//! - **5.10**: Reduce-scatter, `MPI_Reduce_scatter_block()`, `MPI_Reduce_scatter()`
//! - **5.11**: Scan, `MPI_Scan()`, `MPI_Exscan()`
//! - **5.12**: Nonblocking collective operations, `MPI_Ibarrier()`, `MPI_Ibcast()`,
//! `MPI_Igather()`, `MPI_Igatherv()`, `MPI_Iscatter()`, `MPI_Iscatterv()`, `MPI_Iallgather()`,
//! `MPI_Iallgatherv()`, `MPI_Ialltoall()`, `MPI_Ialltoallv()`, `MPI_Ialltoallw()`,
//! `MPI_Ireduce()`, `MPI_Iallreduce()`, `MPI_Ireduce_scatter_block()`, `MPI_Ireduce_scatter()`,
//! `MPI_Iscan()`, `MPI_Iexscan()`

use std::{ptr};

use ffi;
use topology::{Rank, Identifier};
use topology::traits::*;
use datatype::traits::*;

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
    /// See `examples/barrier.rs`
    fn barrier(&self);
}

impl<C: RawCommunicator> Barrier for C {
    fn barrier(&self) {
        unsafe { ffi::MPI_Barrier(self.raw()); }
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
    /// See `examples/broadcast.rs`
    fn broadcast_into<Buf: BufferMut + ?Sized>(&self, buffer: &mut Buf);
}

impl<T: Root> BroadcastInto for T {
    fn broadcast_into<Buf: BufferMut + ?Sized>(&self, buffer: &mut Buf) {
        unsafe {
            ffi::MPI_Bcast(buffer.pointer_mut(), buffer.count(), buffer.datatype().raw(),
                self.root_rank(), self.communicator().raw());
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
    /// See `examples/gather.rs`
    fn gather_into<S: Buffer + ?Sized, R: BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: Option<&mut R>);
}

impl<T: Root> GatherInto for T {
    fn gather_into<S: Buffer + ?Sized, R: BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: Option<&mut R>) {
        unsafe {
            let (recvptr, recvcount, recvtype) = recvbuf.map_or(
                (ptr::null_mut(), 0, u8::equivalent_datatype().raw()),
                |x| (x.pointer_mut(), x.count() / self.communicator().size(), x.datatype().raw()));

            ffi::MPI_Gather(sendbuf.pointer(), sendbuf.count(), sendbuf.datatype().raw(),
                recvptr, recvcount, recvtype, self.root_rank(), self.communicator().raw());
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
    /// See `examples/all_gather.rs`
    fn all_gather_into<S: Buffer + ?Sized, R: BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: &mut R);
}

impl<C: Communicator> AllGatherInto for C {
    fn all_gather_into<S: Buffer + ?Sized, R: BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: &mut R) {
        unsafe {
            ffi::MPI_Allgather(sendbuf.pointer(), sendbuf.count(), sendbuf.datatype().raw(),
                recvbuf.pointer_mut(), recvbuf.count() / self.communicator().size(),
                recvbuf.datatype().raw(), self.communicator().raw());
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
    /// See `examples/scatter.rs`
    fn scatter_into<S: Buffer + ?Sized, R: BufferMut + ?Sized>(&self, sendbuf: Option<&S>, recvbuf: &mut R);
}

impl<T: Root> ScatterInto for T {
    fn scatter_into<S: Buffer + ?Sized, R: BufferMut + ?Sized>(&self, sendbuf: Option<&S>, recvbuf: &mut R) {
        unsafe {
            let (sendptr, sendcount, sendtype) = sendbuf.map_or(
                (ptr::null(), 0, u8::equivalent_datatype().raw()),
                |x| (x.pointer(), x.count() / self.communicator().size(), x.datatype().raw()));

            ffi::MPI_Scatter(sendptr, sendcount, sendtype,
                recvbuf.pointer_mut(), recvbuf.count(), recvbuf.datatype().raw(),
                self.root_rank(), self.communicator().raw());
        }
    }
}
