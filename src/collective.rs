//! Collective communication
//!
//! Developing...
//!
//! # Unfinished features
//!
//! - **5.8**: All-to-all, `MPI_Alltoallw()`
//! - **5.9**: Global reduction operations, `MPI_Op_create()`, `MPI_Op_free()`,
//! `MPI_Op_commutative()`
//! - **5.10**: Reduce-scatter, `MPI_Reduce_scatter()`
//! - **5.12**: Nonblocking collective operations, `MPI_Igatherv()`, `MPI_Iscatterv()`,
//! `MPI_Iallgatherv()`, `MPI_Ialltoallv()`, `MPI_Ialltoallw()`, `MPI_Ireduce_scatter()`

use std::{mem, ptr};

use ffi;
use ffi::{MPI_Request, MPI_Op};

use datatype::traits::*;
use raw::traits::*;
use request::{PlainRequest, ReadRequest, WriteRequest, ReadWriteRequest};
use topology::traits::*;
use topology::{Rank, ProcessIdentifier};

/// Collective communication traits
pub mod traits {
    pub use super::{CommunicatorCollectives, Root, Operation};
}

/// Collective communication patterns defined on `Communicator`s
pub trait CommunicatorCollectives: Communicator {
    /// Barrier synchronization among all processes in a `Communicator`
    ///
    /// Partake in a barrier synchronization across all processes in the `Communicator` `&self`.
    ///
    /// Calling processes (or threads within the calling processes) will enter the barrier and block
    /// execution until all processes in the `Communicator` `&self` have entered the barrier.
    ///
    /// # Examples
    ///
    /// See `examples/barrier.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.3
    fn barrier(&self) {
        unsafe {
            ffi::MPI_Barrier(self.as_raw());
        }
    }

    /// Gather contents of buffers on all participating processes.
    ///
    /// After the call completes, the contents of the send `Buffer`s on all processes will be
    /// concatenated into the receive `Buffer`s on all ranks.
    ///
    /// All send `Buffer`s must contain the same count of elements.
    ///
    /// # Examples
    ///
    /// See `examples/all_gather.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.7
    fn all_gather_into<S: ?Sized, R: ?Sized>(&self, sendbuf: &S, recvbuf: &mut R)
        where S: Buffer,
              R: BufferMut
    {
        unsafe {
            ffi::MPI_Allgather(sendbuf.pointer(),
                               sendbuf.count(),
                               sendbuf.as_datatype().as_raw(),
                               recvbuf.pointer_mut(),
                               recvbuf.count() / self.size(),
                               recvbuf.as_datatype().as_raw(),
                               self.as_raw());
        }
    }

    /// Gather contents of buffers on all participating processes.
    ///
    /// After the call completes, the contents of the send `Buffer`s on all processes will be
    /// concatenated into the receive `Buffer`s on all ranks.
    ///
    /// The send `Buffer`s may contain different counts of elements on different processes. The
    /// distribution of elements in the receive `Buffer`s is specified via `Partitioned`.
    ///
    /// # Examples
    ///
    /// See `examples/all_gather_varcount.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.7
    fn all_gather_varcount_into<S: ?Sized, R: ?Sized>(&self, sendbuf: &S, recvbuf: &mut R)
        where S: Buffer,
              R: PartitionedBufferMut
    {
        unsafe {
            ffi::MPI_Allgatherv(sendbuf.pointer(),
                                sendbuf.count(),
                                sendbuf.as_datatype().as_raw(),
                                recvbuf.pointer_mut(),
                                recvbuf.counts_ptr(),
                                recvbuf.displs_ptr(),
                                recvbuf.as_datatype().as_raw(),
                                self.as_raw());
        }
    }

    /// Distribute the send `Buffer`s from all processes to the receive `Buffer`s on all processes.
    ///
    /// Each process sends and receives the same count of elements to and from each process.
    ///
    /// # Examples
    ///
    /// See `examples/all_to_all.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.8
    fn all_to_all_into<S: ?Sized, R: ?Sized>(&self, sendbuf: &S, recvbuf: &mut R)
        where S: Buffer,
              R: BufferMut
    {
        let c_size = self.size();
        unsafe {
            ffi::MPI_Alltoall(sendbuf.pointer(),
                              sendbuf.count() / c_size,
                              sendbuf.as_datatype().as_raw(),
                              recvbuf.pointer_mut(),
                              recvbuf.count() / c_size,
                              recvbuf.as_datatype().as_raw(),
                              self.as_raw());
        }
    }

    /// Distribute the send `Buffer`s from all processes to the receive `Buffer`s on all processes.
    ///
    /// The count of elements to send and receive to and from each process can vary and is specified
    /// using `Partitioned`.
    ///
    /// # Standard section(s)
    ///
    /// 5.8
    fn all_to_all_varcount_into<S: ?Sized, R: ?Sized>(&self, sendbuf: &S, recvbuf: &mut R)
        where S: PartitionedBuffer,
              R: PartitionedBufferMut
    {
        unsafe {
            ffi::MPI_Alltoallv(sendbuf.pointer(),
                               sendbuf.counts_ptr(),
                               sendbuf.displs_ptr(),
                               sendbuf.as_datatype().as_raw(),
                               recvbuf.pointer_mut(),
                               recvbuf.counts_ptr(),
                               recvbuf.displs_ptr(),
                               recvbuf.as_datatype().as_raw(),
                               self.as_raw());
        }
    }

    /// Performs a global reduction under the operation `op` of the input data in `sendbuf` and
    /// stores the result in `recvbuf` on all processes.
    ///
    /// # Examples
    ///
    /// See `examples/reduce.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.9.6
    fn all_reduce_into<S: ?Sized, R: ?Sized, O>(&self, sendbuf: &S, recvbuf: &mut R, op: &O)
        where S: Buffer,
              R: BufferMut,
              O: Operation
    {
        unsafe {
            ffi::MPI_Allreduce(sendbuf.pointer(),
                               recvbuf.pointer_mut(),
                               sendbuf.count(),
                               sendbuf.as_datatype().as_raw(),
                               op.as_raw(),
                               self.as_raw());
        }
    }

    /// Performs an element-wise global reduction under the operation `op` of the input data in
    /// `sendbuf` and scatters the result into equal sized blocks in the receive buffers on all
    /// processes.
    ///
    /// # Examples
    ///
    /// See `examples/reduce.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.10.1
    fn reduce_scatter_block_into<S: ?Sized, R: ?Sized, O>(&self,
                                                          sendbuf: &S,
                                                          recvbuf: &mut R,
                                                          op: &O)
        where S: Buffer,
              R: BufferMut,
              O: Operation
    {
        assert_eq!(recvbuf.count() * self.size(), sendbuf.count());
        unsafe {
            ffi::MPI_Reduce_scatter_block(sendbuf.pointer(),
                                          recvbuf.pointer_mut(),
                                          recvbuf.count(),
                                          sendbuf.as_datatype().as_raw(),
                                          op.as_raw(),
                                          self.as_raw());
        }
    }

    /// Performs a global inclusive prefix reduction of the data in `sendbuf` into `recvbuf` under
    /// operation `op`.
    ///
    /// # Examples
    ///
    /// See `examples/scan.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.11.1
    fn scan_into<S: ?Sized, R: ?Sized, O>(&self, sendbuf: &S, recvbuf: &mut R, op: &O)
        where S: Buffer,
              R: BufferMut,
              O: Operation
    {
        unsafe {
            ffi::MPI_Scan(sendbuf.pointer(),
                          recvbuf.pointer_mut(),
                          sendbuf.count(),
                          sendbuf.as_datatype().as_raw(),
                          op.as_raw(),
                          self.as_raw());
        }
    }

    /// Performs a global exclusive prefix reduction of the data in `sendbuf` into `recvbuf` under
    /// operation `op`.
    ///
    /// # Examples
    ///
    /// See `examples/scan.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.11.2
    fn exclusive_scan_into<S: ?Sized, R: ?Sized, O>(&self, sendbuf: &S, recvbuf: &mut R, op: &O)
        where S: Buffer,
              R: BufferMut,
              O: Operation
    {
        unsafe {
            ffi::MPI_Exscan(sendbuf.pointer(),
                            recvbuf.pointer_mut(),
                            sendbuf.count(),
                            sendbuf.as_datatype().as_raw(),
                            op.as_raw(),
                            self.as_raw());
        }
    }

    /// Non-blocking barrier synchronization among all processes in a `Communicator`
    ///
    /// Calling processes (or threads within the calling processes) enter the barrier. Completion
    /// methods on the associated request object will block until all processes have entered.
    ///
    /// # Examples
    ///
    /// See `examples/immediate_barrier.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.12.1
    fn immediate_barrier(&self) -> BarrierRequest {
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Ibarrier(self.as_raw(), &mut request);
        }
        BarrierRequest::from_raw(request)
    }

    /// Initiate non-blocking gather of the contents of all `sendbuf`s into all `rcevbuf`s on all
    /// processes in the communicator.
    ///
    /// # Examples
    ///
    /// See `examples/immediate_all_gather.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.12.5
    fn immediate_all_gather_into<'s, 'r, S: ?Sized, R: ?Sized>(&self,
                                                               sendbuf: &'s S,
                                                               recvbuf: &'r mut R)
                                                               -> AllGatherRequest<'s, 'r, S, R>
        where S: 's + Buffer,
              R: 'r + BufferMut
    {
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            let recvcount = recvbuf.count() / self.size();
            ffi::MPI_Iallgather(sendbuf.pointer(),
                                sendbuf.count(),
                                sendbuf.as_datatype().as_raw(),
                                recvbuf.pointer_mut(),
                                recvcount,
                                recvbuf.as_datatype().as_raw(),
                                self.as_raw(),
                                &mut request);
        }
        AllGatherRequest::from_raw(request, sendbuf, recvbuf)
    }

    /// Initiate non-blocking all-to-all communication.
    ///
    /// # Examples
    ///
    /// See `examples/immediate_all_to_all.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.12.6
    fn immediate_all_to_all_into<'s, 'r, S: ?Sized, R: ?Sized>(&self,
                                                               sendbuf: &'s S,
                                                               recvbuf: &'r mut R)
                                                               -> AllToAllRequest<'s, 'r, S, R>
        where S: 's + Buffer,
              R: 'r + BufferMut
    {
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        let c_size = self.size();
        unsafe {
            ffi::MPI_Ialltoall(sendbuf.pointer(),
                               sendbuf.count() / c_size,
                               sendbuf.as_datatype().as_raw(),
                               recvbuf.pointer_mut(),
                               recvbuf.count() / c_size,
                               recvbuf.as_datatype().as_raw(),
                               self.as_raw(),
                               &mut request);
        }
        AllToAllRequest::from_raw(request, sendbuf, recvbuf)
    }

    /// Initiates a non-blocking global reduction under the operation `op` of the input data in
    /// `sendbuf` and stores the result in `recvbuf` on all processes.
    ///
    /// # Standard section(s)
    ///
    /// 5.12.8
    fn immediate_all_reduce_into<'s, 'r, S: ?Sized, R: ?Sized, O>
        (&self,
         sendbuf: &'s S,
         recvbuf: &'r mut R,
         op: &O)
         -> AllReduceRequest<'s, 'r, S, R>
        where S: 's + Buffer,
              R: 'r + BufferMut,
              O: Operation
    {
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Iallreduce(sendbuf.pointer(),
                                recvbuf.pointer_mut(),
                                sendbuf.count(),
                                sendbuf.as_datatype().as_raw(),
                                op.as_raw(),
                                self.as_raw(),
                                &mut request);
        }
        AllReduceRequest::from_raw(request, sendbuf, recvbuf)
    }

    /// Initiates a non-blocking element-wise global reduction under the operation `op` of the
    /// input data in `sendbuf` and scatters the result into equal sized blocks in the receive
    /// buffers on all processes.
    ///
    /// # Standard section(s)
    ///
    /// 5.12.9
    fn immediate_reduce_scatter_block_into<'s, 'r, S: ?Sized, R: ?Sized, O>
        (&self,
         sendbuf: &'s S,
         recvbuf: &'r mut R,
         op: &O)
         -> ReduceScatterBlockRequest<'s, 'r, S, R>
        where S: 's + Buffer,
              R: 'r + BufferMut,
              O: Operation
    {
        assert_eq!(recvbuf.count() * self.size(), sendbuf.count());
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Ireduce_scatter_block(sendbuf.pointer(),
                                           recvbuf.pointer_mut(),
                                           recvbuf.count(),
                                           sendbuf.as_datatype().as_raw(),
                                           op.as_raw(),
                                           self.as_raw(),
                                           &mut request);
        }
        ReduceScatterBlockRequest::from_raw(request, sendbuf, recvbuf)
    }

    /// Initiates a non-blocking global inclusive prefix reduction of the data in `sendbuf` into
    /// `recvbuf` under operation `op`.
    ///
    /// # Standard section(s)
    ///
    /// 5.12.11
    fn immediate_scan_into<'s, 'r, S: ?Sized, R: ?Sized, O>(&self,
                                                            sendbuf: &'s S,
                                                            recvbuf: &'r mut R,
                                                            op: &O)
                                                            -> ScanRequest<'s, 'r, S, R>
        where S: 's + Buffer,
              R: 'r + BufferMut,
              O: Operation
    {
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Iscan(sendbuf.pointer(),
                           recvbuf.pointer_mut(),
                           sendbuf.count(),
                           sendbuf.as_datatype().as_raw(),
                           op.as_raw(),
                           self.as_raw(),
                           &mut request);
        }
        ScanRequest::from_raw(request, sendbuf, recvbuf)
    }

    /// Initiates a non-blocking global exclusive prefix reduction of the data in `sendbuf` into
    /// `recvbuf` under operation `op`.
    ///
    /// # Standard section(s)
    ///
    /// 5.12.12
    fn immediate_exclusive_scan_into<'s, 'r, S: ?Sized, R: ?Sized, O>
        (&self,
         sendbuf: &'s S,
         recvbuf: &'r mut R,
         op: &O)
         -> ExclusiveScanRequest<'s, 'r, S, R>
        where S: 's + Buffer,
              R: 'r + BufferMut,
              O: Operation
    {
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Iexscan(sendbuf.pointer(),
                             recvbuf.pointer_mut(),
                             sendbuf.count(),
                             sendbuf.as_datatype().as_raw(),
                             op.as_raw(),
                             self.as_raw(),
                             &mut request);
        }
        ExclusiveScanRequest::from_raw(request, sendbuf, recvbuf)
    }
}

impl<C: Communicator> CommunicatorCollectives for C {}

/// Something that can take the role of 'root' in a collective operation.
///
/// Many collective operations define a 'root' process that takes a special role in the
/// communication. These collective operations are implemented as default methods of this trait.
pub trait Root: AsCommunicator
{
    /// Rank of the root process
    fn root_rank(&self) -> Rank;

    /// Broadcast of the contents of a buffer
    ///
    /// After the call completes, the `Buffer` on all processes in the `Communicator` of the `Root`
    /// `&self` will contain what it contains on the `Root`.
    ///
    /// # Examples
    ///
    /// See `examples/broadcast.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.4
    fn broadcast_into<Buf: ?Sized>(&self, buffer: &mut Buf)
        where Buf: BufferMut
    {
        unsafe {
            ffi::MPI_Bcast(buffer.pointer_mut(),
                           buffer.count(),
                           buffer.as_datatype().as_raw(),
                           self.root_rank(),
                           self.as_communicator().as_raw());
        }
    }

    /// Gather contents of buffers on `Root`.
    ///
    /// After the call completes, the contents of the `Buffer`s on all ranks will be
    /// concatenated into the `Buffer` on `Root`.
    ///
    /// All send `Buffer`s must have the same count of elements.
    ///
    /// This function must be called on all non-root processes.
    ///
    /// # Examples
    ///
    /// See `examples/gather.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.5
    fn gather_into<S: ?Sized>(&self, sendbuf: &S)
        where S: Buffer
    {
        assert!(self.as_communicator().rank() != self.root_rank());
        unsafe {
            ffi::MPI_Gather(sendbuf.pointer(),
                            sendbuf.count(),
                            sendbuf.as_datatype().as_raw(),
                            ptr::null_mut(),
                            0,
                            u8::equivalent_datatype().as_raw(),
                            self.root_rank(),
                            self.as_communicator().as_raw());
        }
    }

    /// Gather contents of buffers on `Root`.
    ///
    /// After the call completes, the contents of the `Buffer`s on all ranks will be
    /// concatenated into the `Buffer` on `Root`.
    ///
    /// All send `Buffer`s must have the same count of elements.
    ///
    /// This function must be called on the root process.
    ///
    /// # Examples
    ///
    /// See `examples/gather.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.5
    fn gather_into_root<S: ?Sized, R: ?Sized>(&self, sendbuf: &S, recvbuf: &mut R)
        where S: Buffer,
              R: BufferMut
    {
        assert!(self.as_communicator().rank() == self.root_rank());
        unsafe {
            let recvcount = recvbuf.count() / self.as_communicator().size();
            ffi::MPI_Gather(sendbuf.pointer(),
                            sendbuf.count(),
                            sendbuf.as_datatype().as_raw(),
                            recvbuf.pointer_mut(),
                            recvcount,
                            recvbuf.as_datatype().as_raw(),
                            self.root_rank(),
                            self.as_communicator().as_raw());
        }
    }

    /// Gather contents of buffers on `Root`.
    ///
    /// After the call completes, the contents of the `Buffer`s on all ranks will be
    /// concatenated into the `Buffer` on `Root`.
    ///
    /// The send `Buffer`s may contain different counts of elements on different processes. The
    /// distribution of elements in the receive `Buffer` is specified via `Partitioned`.
    ///
    /// This function must be called on all non-root processes.
    ///
    /// # Examples
    ///
    /// See `examples/gather_varcount.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.5
    fn gather_varcount_into<S: ?Sized>(&self, sendbuf: &S)
        where S: Buffer
    {
        assert!(self.as_communicator().rank() != self.root_rank());
        unsafe {
            ffi::MPI_Gatherv(sendbuf.pointer(),
                             sendbuf.count(),
                             sendbuf.as_datatype().as_raw(),
                             ptr::null_mut(),
                             ptr::null(),
                             ptr::null(),
                             u8::equivalent_datatype().as_raw(),
                             self.root_rank(),
                             self.as_communicator().as_raw());
        }
    }

    /// Gather contents of buffers on `Root`.
    ///
    /// After the call completes, the contents of the `Buffer`s on all ranks will be
    /// concatenated into the `Buffer` on `Root`.
    ///
    /// The send `Buffer`s may contain different counts of elements on different processes. The
    /// distribution of elements in the receive `Buffer` is specified via `Partitioned`.
    ///
    /// This function must be called on the root process.
    ///
    /// # Examples
    ///
    /// See `examples/gather_varcount.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.5
    fn gather_varcount_into_root<S: ?Sized, R: ?Sized>(&self, sendbuf: &S, recvbuf: &mut R)
        where S: Buffer,
              R: PartitionedBufferMut
    {
        assert!(self.as_communicator().rank() == self.root_rank());
        unsafe {
            ffi::MPI_Gatherv(sendbuf.pointer(),
                             sendbuf.count(),
                             sendbuf.as_datatype().as_raw(),
                             recvbuf.pointer_mut(),
                             recvbuf.counts_ptr(),
                             recvbuf.displs_ptr(),
                             recvbuf.as_datatype().as_raw(),
                             self.root_rank(),
                             self.as_communicator().as_raw());
        }
    }

    /// Scatter contents of a buffer on the root process to all processes.
    ///
    /// After the call completes each participating process will have received a part of the send
    /// `Buffer` on the root process.
    ///
    /// All send `Buffer`s must have the same count of elements.
    ///
    /// This function must be called on all non-root processes.
    ///
    /// # Examples
    ///
    /// See `examples/scatter.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.6
    fn scatter_into<R: ?Sized>(&self, recvbuf: &mut R)
        where R: BufferMut
    {
        assert!(self.as_communicator().rank() != self.root_rank());
        unsafe {
            ffi::MPI_Scatter(ptr::null(),
                             0,
                             u8::equivalent_datatype().as_raw(),
                             recvbuf.pointer_mut(),
                             recvbuf.count(),
                             recvbuf.as_datatype().as_raw(),
                             self.root_rank(),
                             self.as_communicator().as_raw());
        }
    }

    /// Scatter contents of a buffer on the root process to all processes.
    ///
    /// After the call completes each participating process will have received a part of the send
    /// `Buffer` on the root process.
    ///
    /// All send `Buffer`s must have the same count of elements.
    ///
    /// This function must be called on the root process.
    ///
    /// # Examples
    ///
    /// See `examples/scatter.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.6
    ///
    /// # Examples
    ///
    /// See `examples/scatter.rs`
    fn scatter_into_root<S: ?Sized, R: ?Sized>(&self, sendbuf: &S, recvbuf: &mut R)
        where S: Buffer,
              R: BufferMut
    {
        assert!(self.as_communicator().rank() == self.root_rank());
        let sendcount = sendbuf.count() / self.as_communicator().size();
        unsafe {
            ffi::MPI_Scatter(sendbuf.pointer(),
                             sendcount,
                             sendbuf.as_datatype().as_raw(),
                             recvbuf.pointer_mut(),
                             recvbuf.count(),
                             recvbuf.as_datatype().as_raw(),
                             self.root_rank(),
                             self.as_communicator().as_raw());
        }
    }

    /// Scatter contents of a buffer on the root process to all processes.
    ///
    /// After the call completes each participating process will have received a part of the send
    /// `Buffer` on the root process.
    ///
    /// The send `Buffer` may contain different counts of elements for different processes. The
    /// distribution of elements in the send `Buffer` is specified via `Partitioned`.
    ///
    /// This function must be called on all non-root processes.
    ///
    /// # Examples
    ///
    /// See `examples/scatter_varcount.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.6
    fn scatter_varcount_into<R: ?Sized>(&self, recvbuf: &mut R)
        where R: BufferMut
    {
        assert!(self.as_communicator().rank() != self.root_rank());
        unsafe {
            ffi::MPI_Scatterv(ptr::null(),
                              ptr::null(),
                              ptr::null(),
                              u8::equivalent_datatype().as_raw(),
                              recvbuf.pointer_mut(),
                              recvbuf.count(),
                              recvbuf.as_datatype().as_raw(),
                              self.root_rank(),
                              self.as_communicator().as_raw());
        }
    }

    /// Scatter contents of a buffer on the root process to all processes.
    ///
    /// After the call completes each participating process will have received a part of the send
    /// `Buffer` on the root process.
    ///
    /// The send `Buffer` may contain different counts of elements for different processes. The
    /// distribution of elements in the send `Buffer` is specified via `Partitioned`.
    ///
    /// This function must be called on the root process.
    ///
    /// # Examples
    ///
    /// See `examples/scatter_varcount.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.6
    fn scatter_varcount_into_root<S: ?Sized, R: ?Sized>(&self, sendbuf: &S, recvbuf: &mut R)
        where S: PartitionedBuffer,
              R: BufferMut
    {
        assert!(self.as_communicator().rank() == self.root_rank());
        unsafe {
            ffi::MPI_Scatterv(sendbuf.pointer(),
                              sendbuf.counts_ptr(),
                              sendbuf.displs_ptr(),
                              sendbuf.as_datatype().as_raw(),
                              recvbuf.pointer_mut(),
                              recvbuf.count(),
                              recvbuf.as_datatype().as_raw(),
                              self.root_rank(),
                              self.as_communicator().as_raw());
        }
    }

    /// Performs a global reduction under the operation `op` of the input data in `sendbuf` and
    /// stores the result on the `Root` process.
    ///
    /// This function must be called on all non-root processes.
    ///
    /// # Examples
    ///
    /// See `examples/reduce.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.9.1
    fn reduce_into<S: ?Sized, O>(&self, sendbuf: &S, op: &O)
        where S: Buffer,
              O: Operation
    {
        assert!(self.as_communicator().rank() != self.root_rank());
        unsafe {
            ffi::MPI_Reduce(sendbuf.pointer(),
                            ptr::null_mut(),
                            sendbuf.count(),
                            sendbuf.as_datatype().as_raw(),
                            op.as_raw(),
                            self.root_rank(),
                            self.as_communicator().as_raw());
        }
    }

    /// Performs a global reduction under the operation `op` of the input data in `sendbuf` and
    /// stores the result on the `Root` process.
    ///
    /// This function must be called on the root process.
    ///
    /// # Examples
    ///
    /// See `examples/reduce.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.9.1
    fn reduce_into_root<S: ?Sized, R: ?Sized, O>(&self, sendbuf: &S, recvbuf: &mut R, op: &O)
        where S: Buffer,
              R: BufferMut,
              O: Operation
    {
        assert!(self.as_communicator().rank() == self.root_rank());
        unsafe {
            ffi::MPI_Reduce(sendbuf.pointer(),
                            recvbuf.pointer_mut(),
                            sendbuf.count(),
                            sendbuf.as_datatype().as_raw(),
                            op.as_raw(),
                            self.root_rank(),
                            self.as_communicator().as_raw());
        }
    }

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

    /// Initiate broadcast of a value from the `Root` process to all other processes.
    ///
    /// # Examples
    ///
    /// See `examples/immediate_broadcast.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.12.2
    fn immediate_broadcast_into<'b, Buf: ?Sized>(&self,
                                                 buf: &'b mut Buf)
                                                 -> BroadcastRequest<'b, Buf>
        where Buf: 'b + BufferMut
    {
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Ibcast(buf.pointer_mut(),
                            buf.count(),
                            buf.as_datatype().as_raw(),
                            self.root_rank(),
                            self.as_communicator().as_raw(),
                            &mut request);
        }
        BroadcastRequest::from_raw(request, buf)
    }

    /// Initiate non-blocking gather of the contents of all `sendbuf`s on `Root` `&self`.
    ///
    /// This function must be called on all non-root processes.
    ///
    /// # Examples
    ///
    /// See `examples/immediate_gather.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.12.3
    fn immediate_gather_into<'s, S: ?Sized>(&self, sendbuf: &'s S) -> GatherRequest<'s, S>
        where S: 's + Buffer
    {
        assert!(self.as_communicator().rank() != self.root_rank());
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Igather(sendbuf.pointer(),
                             sendbuf.count(),
                             sendbuf.as_datatype().as_raw(),
                             ptr::null_mut(),
                             0,
                             u8::equivalent_datatype().as_raw(),
                             self.root_rank(),
                             self.as_communicator().as_raw(),
                             &mut request);
        }
        GatherRequest::from_raw(request, sendbuf)
    }

    /// Initiate non-blocking gather of the contents of all `sendbuf`s on `Root` `&self`.
    ///
    /// This function must be called on the root processes.
    ///
    /// # Examples
    ///
    /// See `examples/immediate_gather.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.12.3
    fn immediate_gather_into_root<'s, 'r, S: ?Sized, R: ?Sized>
        (&self,
         sendbuf: &'s S,
         recvbuf: &'r mut R)
         -> GatherRootRequest<'s, 'r, S, R>
        where S: 's + Buffer,
              R: 'r + BufferMut
    {
        assert!(self.as_communicator().rank() == self.root_rank());
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            let recvcount = recvbuf.count() / self.as_communicator().size();
            ffi::MPI_Igather(sendbuf.pointer(),
                             sendbuf.count(),
                             sendbuf.as_datatype().as_raw(),
                             recvbuf.pointer_mut(),
                             recvcount,
                             recvbuf.as_datatype().as_raw(),
                             self.root_rank(),
                             self.as_communicator().as_raw(),
                             &mut request);
        }
        GatherRootRequest::from_raw(request, sendbuf, recvbuf)
    }

    /// Initiate non-blocking scatter of the contents of `sendbuf` from `Root` `&self`.
    ///
    /// This function must be called on all non-root processes.
    ///
    /// # Examples
    ///
    /// See `examples/immediate_scatter.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.12.4
    fn immediate_scatter_into<'r, R: ?Sized>(&self, recvbuf: &'r mut R) -> ScatterRequest<'r, R>
        where R: 'r + BufferMut
    {
        assert!(self.as_communicator().rank() != self.root_rank());
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Iscatter(ptr::null(),
                              0,
                              u8::equivalent_datatype().as_raw(),
                              recvbuf.pointer_mut(),
                              recvbuf.count(),
                              recvbuf.as_datatype().as_raw(),
                              self.root_rank(),
                              self.as_communicator().as_raw(),
                              &mut request);
        }
        ScatterRequest::from_raw(request, recvbuf)
    }

    /// Initiate non-blocking scatter of the contents of `sendbuf` from `Root` `&self`.
    ///
    /// This function must be called on the root processes.
    ///
    /// # Examples
    ///
    /// See `examples/immediate_scatter.rs`
    ///
    /// # Standard section(s)
    ///
    /// 5.12.4
    fn immediate_scatter_into_root<'s, 'r, S: ?Sized, R: ?Sized>
        (&self,
         sendbuf: &'s S,
         recvbuf: &'r mut R)
         -> ScatterRootRequest<'s, 'r, S, R>
        where S: 's + Buffer,
              R: 'r + BufferMut
    {
        assert!(self.as_communicator().rank() == self.root_rank());
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            let sendcount = sendbuf.count() / self.as_communicator().size();
            ffi::MPI_Iscatter(sendbuf.pointer(),
                              sendcount,
                              sendbuf.as_datatype().as_raw(),
                              recvbuf.pointer_mut(),
                              recvbuf.count(),
                              recvbuf.as_datatype().as_raw(),
                              self.root_rank(),
                              self.as_communicator().as_raw(),
                              &mut request);
        }
        ScatterRootRequest::from_raw(request, sendbuf, recvbuf)
    }

    /// Initiates a non-blacking global reduction under the operation `op` of the input data in
    /// `sendbuf` and stores the result on the `Root` process.
    ///
    /// This function must be called on all non-root processes.
    ///
    /// # Standard section(s)
    ///
    /// 5.12.7
    fn immediate_reduce_into<'s, S: ?Sized, O>(&self,
                                               sendbuf: &'s S,
                                               op: &O)
                                               -> ReduceRequest<'s, S>
        where S: 's + Buffer,
              O: Operation
    {
        assert!(self.as_communicator().rank() != self.root_rank());
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Ireduce(sendbuf.pointer(),
                             ptr::null_mut(),
                             sendbuf.count(),
                             sendbuf.as_datatype().as_raw(),
                             op.as_raw(),
                             self.root_rank(),
                             self.as_communicator().as_raw(),
                             &mut request);
        }
        ReduceRequest::from_raw(request, sendbuf)
    }

    /// Initiates a non-blocking global reduction under the operation `op` of the input data in
    /// `sendbuf` and stores the result on the `Root` process.
    ///
    /// This function must be called on the root process.
    ///
    /// # Standard section(s)
    ///
    /// 5.12.7
    fn immediate_reduce_into_root<'s, 'r, S: ?Sized, R: ?Sized, O>
        (&self,
         sendbuf: &'s S,
         recvbuf: &'r mut R,
         op: &O)
         -> ReduceRootRequest<'s, 'r, S, R>
        where S: 's + Buffer,
              R: 'r + BufferMut,
              O: Operation
    {
        assert!(self.as_communicator().rank() == self.root_rank());
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Ireduce(sendbuf.pointer(),
                             recvbuf.pointer_mut(),
                             sendbuf.count(),
                             sendbuf.as_datatype().as_raw(),
                             op.as_raw(),
                             self.root_rank(),
                             self.as_communicator().as_raw(),
                             &mut request);
        }
        ReduceRootRequest::from_raw(request, sendbuf, recvbuf)
    }
}

impl<'a, C: 'a + Communicator> Root for ProcessIdentifier<'a, C> {
    fn root_rank(&self) -> Rank {
        self.rank()
    }
}

/// An operation to be used in a reduction or scan type operation, e.g. `MPI_SUM`
pub trait Operation: AsRaw<Raw = MPI_Op> { }
impl<'a, T: 'a + Operation> Operation for &'a T {}

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
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Operation for SystemOperation {}

macro_rules! reduce_into_specializations {
    ($($name:ident => $operation:expr),*) => (
        $(fn $name<S: Buffer + ?Sized, R: BufferMut + ?Sized>(&self, sendbuf: &S, recvbuf: Option<&mut R>) {
            self.reduce_into(sendbuf, recvbuf, $operation)
        })*
    )
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
pub fn reduce_local_into<S: ?Sized, R: ?Sized, O>(inbuf: &S, inoutbuf: &mut R, op: &O)
    where S: Buffer,
          R: BufferMut,
          O: Operation
{
    unsafe {
        ffi::MPI_Reduce_local(inbuf.pointer(),
                              inoutbuf.pointer_mut(),
                              inbuf.count(),
                              inbuf.as_datatype().as_raw(),
                              op.as_raw());
    }
}

/// A request object for an immediate (non-blocking) barrier operation
pub type BarrierRequest = PlainRequest;

/// A request object for an immediate (non-blocking) broadcast operation
pub type BroadcastRequest<'b, Buf> = WriteRequest<'b, Buf>;

/// A request object for an immediate (non-blocking) gather operation
pub type GatherRequest<'s, S> = ReadRequest<'s, S>;

/// A request object for an immediate (non-blocking) gather operation on the root process
pub type GatherRootRequest<'s, 'r, S, R> = ReadWriteRequest<'s, 'r, S, R>;

/// A request object for an immediate (non-blocking) all-gather operation
pub type AllGatherRequest<'s, 'r, S, R> = ReadWriteRequest<'s, 'r, S, R>;

/// A request object for an immediate (non-blocking) scatter operation
pub type ScatterRequest<'r, R> = WriteRequest<'r, R>;

/// A request object for an immediate (non-blocking) scatter operation on the root process
pub type ScatterRootRequest<'s, 'r, S, R> = ReadWriteRequest<'s, 'r, S, R>;

/// A request object for an immediate (non-blocking) all-to-all operation
pub type AllToAllRequest<'s, 'r, S, R> = ReadWriteRequest<'s, 'r, S, R>;

/// A request object for an immediat (non-blocking) reduce operation
pub type ReduceRequest<'s, S> = ReadRequest<'s, S>;

/// A request object for an immediat (non-blocking) reduce operation on the root process
pub type ReduceRootRequest<'s, 'r, S, R> = ReadWriteRequest<'s, 'r, S, R>;

/// A request object for an immediate (non-blocking) all-reduce operation
pub type AllReduceRequest<'s, 'r, S, R> = ReadWriteRequest<'s, 'r, S, R>;

/// A request object for an immediate (non-blocking) reduce-scatter-block operation
pub type ReduceScatterBlockRequest<'s, 'r, S, R> = ReadWriteRequest<'s, 'r, S, R>;

/// A request object for an immediate (non-blocking) scan operation
pub type ScanRequest<'s, 'r, S, R> = ReadWriteRequest<'s, 'r, S, R>;

/// A request object for an immediate (non-blocking) exclusive scan operation
pub type ExclusiveScanRequest<'s, 'r, S, R> = ReadWriteRequest<'s, 'r, S, R>;
