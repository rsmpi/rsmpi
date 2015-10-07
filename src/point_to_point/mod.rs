//! Point to point communication
//!
//! Endpoints of communication are mostly described by types that implement the `Source` and
//! `Destination` trait. Communication operations are implemented as traits that have blanket
//! implementations for these two traits.
//!
//! # Unfinished features
//!
//! - **3.2.6**: `MPI_STATUS_IGNORE`
//! - **3.6**: Buffer usage, `MPI_Buffer_attach()`, `MPI_Buffer_detach()`
//! - **3.7**: Nonblocking mode:
//!   - Completion, `MPI_Waitany()`, `MPI_Waitall()`, `MPI_Waitsome()`,
//!   `MPI_Testany()`, `MPI_Testall()`, `MPI_Testsome()`, `MPI_Request_get_status()`
//! - **3.8**:
//!   - Cancellation, `MPI_Cancel()`, `MPI_Test_cancelled()`
//! - **3.9**: Persistent requests, `MPI_Send_init()`, `MPI_Bsend_init()`, `MPI_Ssend_init()`,
//! `MPI_Rsend_init()`, `MPI_Recv_init()`, `MPI_Start()`, `MPI_Startall()`

use std::{mem, fmt};
use std::marker::PhantomData;

use libc::c_int;

use conv::ConvUtil;

use ffi;
use ffi::{MPI_Status, MPI_Message, MPI_Request};
use topology::{SystemCommunicator, UserCommunicator, Rank, Identifier};
use topology::traits::*;
use datatype::traits::*;
use super::{Error, Count, Tag};

// TODO: rein in _with_tag ugliness, use optional tags or make tag part of Source and Destination

pub mod traits;

/// Something that can be used as the source in a point to point receive operation
///
/// # Examples
///
/// - An `Identifier` used as a source for a receive operation will receive data only from the
/// identified process.
/// - A communicator can also be used as a source. The receive operation will receive data from
/// any process in the communicator.
///
/// # Standard section(s)
///
/// 3.2.3
pub trait Source: Communicator {
    /// `Rank` that identifies the source
    fn source_rank(&self) -> Rank;
}

// TODO: this does not work for now, needs specialization
//impl<C: Communicator> Source for C {
//    fn source_rank(&self) -> Rank {
//        ffi::RSMPI_ANY_SOURCE
//    }
//}

impl Source for SystemCommunicator {
    fn source_rank(&self) -> Rank {
        ffi::RSMPI_ANY_SOURCE
    }
}

impl<'a> Source for &'a UserCommunicator {
    fn source_rank(&self) -> Rank {
        ffi::RSMPI_ANY_SOURCE
    }
}

impl<'a, C: 'a + RawCommunicator> Source for Identifier<'a, C> {
    fn source_rank(&self) -> Rank {
        self.rank()
    }
}

/// Something that can be used as the destination in a point to point send operation
///
/// # Examples
/// - Using an `Identifier` as the destination will send data to that specific process.
///
/// # Standard section(s)
///
/// 3.2.3
pub trait Destination: Communicator {
    /// `Rank` that identifies the destination
    fn destination_rank(&self) -> Rank;
}

impl<'a, C: 'a + RawCommunicator> Destination for Identifier<'a, C> {
    fn destination_rank(&self) -> Rank {
        self.rank()
    }
}

/// Blocking standard mode send operation
///
/// # Examples
///
/// ```no_run
/// use mpi::traits::*;
///
/// let universe = mpi::initialize().unwrap();
/// let world = universe.world();
///
/// let v = vec![ 1.0f64, 2.0, 3.0 ];
/// world.process_at_rank(1).send(&v[..]);
/// ```
///
/// # Standard section(s)
///
/// 3.2.1
pub trait Send {
    /// Send the contents of a `Buffer` to the `Destination` `&self` and tag it.
    fn send_with_tag<Buf: Buffer + ?Sized>(&self, buf: &Buf, tag: Tag);

    /// Send the contents of a `Buffer` to the `Destination` `&self`.
    ///
    /// # Examples
    /// See `examples/send_receive.rs`
    fn send<Buf: Buffer + ?Sized>(&self, buf: &Buf) {
        self.send_with_tag(buf, Tag::default())
    }
}

impl<Dest: Destination> Send for Dest {
    fn send_with_tag<Buf: Buffer + ?Sized>(&self, buf: &Buf, tag: Tag) {
        unsafe {
            ffi::MPI_Send(buf.pointer(), buf.count(), buf.datatype().raw(),
                self.destination_rank(), tag, self.communicator().raw());
        }
    }
}

/// Blocking buffered mode send operation
///
/// # Standard section(s)
///
/// 3.4
pub trait BufferedSend {
    /// Send the contents of a `Buffer` to the `Destination` `&self` and tag it.
    fn buffered_send_with_tag<Buf: Buffer + ?Sized>(&self, buf: &Buf, tag: Tag);

    /// Send the contents of a `Buffer` to the `Destination` `&self`.
    fn buffered_send<Buf: Buffer + ?Sized>(&self, buf: &Buf) {
        self.buffered_send_with_tag(buf, Tag::default())
    }
}

impl<Dest: Destination> BufferedSend for Dest {
    fn buffered_send_with_tag<Buf: Buffer + ?Sized>(&self, buf: &Buf, tag: Tag) {
        unsafe {
            ffi::MPI_Bsend(buf.pointer(), buf.count(), buf.datatype().raw(),
                self.destination_rank(), tag, self.communicator().raw());
        }
    }
}

/// Blocking synchronous mode send operation
///
/// # Standard section(s)
///
/// 3.4
pub trait SynchronousSend {
    /// Send the contents of a `Buffer` to the `Destination` `&self` and tag it.
    ///
    /// Completes only once the matching receive operation has started.
    fn synchronous_send_with_tag<Buf: Buffer + ?Sized>(&self, buf: &Buf, tag: Tag);

    /// Send the contents of a `Buffer` to the `Destination` `&self`.
    ///
    /// Completes only once the matching receive operation has started.
    fn synchronous_send<Buf: Buffer + ?Sized>(&self, buf: &Buf) {
        self.synchronous_send_with_tag(buf, Tag::default())
    }
}

impl<Dest: Destination> SynchronousSend for Dest {
    fn synchronous_send_with_tag<Buf: Buffer + ?Sized>(&self, buf: &Buf, tag: Tag) {
        unsafe {
            ffi::MPI_Ssend(buf.pointer(), buf.count(), buf.datatype().raw(),
                self.destination_rank(), tag, self.communicator().raw());
        }
    }
}

/// Blocking ready mode send operation
///
/// # Standard section(s)
///
/// 3.4
pub trait ReadySend {
    /// Send the contents of a `Buffer` to the `Destination` `&self` and tag it.
    ///
    /// Fails if the matching receive operation has not been posted.
    fn ready_send_with_tag<Buf: Buffer + ?Sized>(&self, buf: &Buf, tag: Tag);

    /// Send the contents of a `Buffer` to the `Destination` `&self`.
    ///
    /// Fails if the matching receive operation has not been posted.
    fn ready_send<Buf: Buffer + ?Sized>(&self, buf: &Buf) {
        self.ready_send_with_tag(buf, Tag::default())
    }
}

impl<Dest: Destination> ReadySend for Dest {
    fn ready_send_with_tag<Buf: Buffer + ?Sized>(&self, buf: &Buf, tag: Tag) {
        unsafe {
            ffi::MPI_Rsend(buf.pointer(), buf.count(), buf.datatype().raw(),
                self.destination_rank(), tag, self.communicator().raw());
        }
    }
}

/// Describes the result of a point to point receive operation.
///
/// # Standard section(s)
///
/// 3.2.5
pub struct Status(MPI_Status);

impl Status {
    /// The rank of the message source
    pub fn source_rank(&self) -> Rank {
        self.0.MPI_SOURCE
    }

    /// The message tag
    pub fn tag(&self) -> Tag {
        self.0.MPI_TAG
    }

    /// An error code
    pub fn error(&self) -> Error {
        self.0.MPI_ERROR
    }

    /// Number of instances of the type contained in the message
    pub fn count<D: RawDatatype>(&self, d: D) -> Count {
        let mut count: Count = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Get_count(&self.0, d.raw(), &mut count as *mut Count) };
        count
    }
}

impl fmt::Debug for Status {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "Status {{ source_rank: {}, tag: {}, error: {} }}",
               self.source_rank(), self.tag(), self.error())
    }
}

/// Probe a source for incoming messages.
///
/// An ordinary `probe()` returns a `Status` which allows inspection of the properties of the
/// incoming message, but does not guarantee reception by a subsequent `receive()` (especially in a
/// multi-threaded set-up). For a probe operation with stronger guarantees, see `MatchedProbe`.
///
/// # Standard section(s)
///
/// 3.8.1
pub trait Probe {
    /// Probe `Source` `&self` for incoming messages with a certain tag.
    fn probe_with_tag(&self, tag: Tag) -> Status;

    /// Probe `Source` `&self` for incoming messages with any tag.
    fn probe(&self) -> Status {
        self.probe_with_tag(ffi::RSMPI_ANY_TAG)
    }
}

impl<Src: Source> Probe for Src {
    fn probe_with_tag(&self, tag: Tag) -> Status {
        let mut status: MPI_Status = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Probe(self.source_rank(), tag, self.communicator().raw(),
                &mut status as *mut MPI_Status);
        };
        Status(status)
    }
}

/// Describes an pending incoming message, probed by a `matched_probe()`.
///
/// # Standard section(s)
///
/// 3.8.2
#[must_use]
pub struct Message(MPI_Message);

impl Message {
    /// True if the `Source` for the probe was the null process.
    pub fn is_no_proc(&self) -> bool {
        unsafe { self.raw() == ffi::RSMPI_MESSAGE_NO_PROC }
    }

    unsafe fn raw(&self) -> MPI_Message {
        self.0
    }

    unsafe fn ptr_mut(&mut self) -> *mut MPI_Message {
        &mut self.0 as *mut MPI_Message
    }
}

impl Drop for Message {
    fn drop(&mut self) {
        unsafe {
            assert!(self.raw() == ffi::RSMPI_MESSAGE_NULL,
                "matched messag dropped without receiving.");
        }
    }
}

/// Probe a source for incoming messages with guaranteed reception.
///
/// A `matched_probe()` returns both a `Status` that describes the properties of a pending incoming
/// message and a `Message` which can and *must* subsequently be used in a `matched_receive()`
/// to receive the probed message.
///
/// # Standard section(s)
///
/// 3.8.2
pub trait MatchedProbe {
    /// Probe `Source` `&self` for incoming messages with a certain tag.
    fn matched_probe_with_tag(&self, tag: Tag) -> (Message, Status);

    /// Probe `Source` `&self` for incoming messages with any tag.
    fn matched_probe(&self) -> (Message, Status) {
        self.matched_probe_with_tag(ffi::RSMPI_ANY_TAG)
    }
}

impl<Src: Source> MatchedProbe for Src {
    fn matched_probe_with_tag(&self, tag: Tag) -> (Message, Status) {
        let mut message: MPI_Message = unsafe { mem::uninitialized() };
        let mut status: MPI_Status = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Mprobe(self.source_rank(), tag, self.communicator().raw(),
                &mut message as *mut MPI_Message, &mut status as *mut MPI_Status);
        }
        (Message(message), Status(status))
    }
}

/// Receive a previously probed message containing a single instance of type `Msg`.
///
/// # Standard section(s)
///
/// 3.8.3
pub trait MatchedReceive {
    /// Receives the message `&self` which contains a single instance of type `Msg` or None if
    /// receiving from the null process.
    fn matched_receive<Msg: EquivalentDatatype>(self) -> (Option<Msg>, Status);
}

impl MatchedReceive for Message {
    fn matched_receive<Msg: EquivalentDatatype>(self) -> (Option<Msg>, Status) {
        let is_no_proc = self.is_no_proc();
        let mut res: Msg = unsafe { mem::uninitialized() };
        let status = self.matched_receive_into(&mut res);
        if is_no_proc {
            (None, status)
        } else {
            (Some(res), status)
        }
    }
}

/// Receive a previously probed message into a `Buffer`.
///
/// # Standard section(s)
///
/// 3.8.3
pub trait MatchedReceiveInto {
    /// Receive the message `&self` with contents matching `buf`.
    ///
    /// Receiving from the null process leaves `buf` untouched.
    fn matched_receive_into<Buf: BufferMut + ?Sized>(self, buf: &mut Buf) -> Status;
}

impl MatchedReceiveInto for Message {
    fn matched_receive_into<Buf: BufferMut + ?Sized>(mut self, buf: &mut Buf) -> Status {
        let mut status: MPI_Status = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Mrecv(buf.pointer_mut(), buf.count(), buf.datatype().raw(),
                self.ptr_mut(), &mut status as *mut MPI_Status);
            assert_eq!(self.raw(), ffi::RSMPI_MESSAGE_NULL);
        }
        Status(status)
    }
}

/// Receive a previously probed message containing multiple instances of type `Msg` into a `Vec`.
///
/// # Standard section(s)
///
/// 3.8.3
pub trait MatchedReceiveVec {
    /// Receives the message `&self` which contains multiple instances of type `Msg` into a `Vec`
    /// or `None` if receiving from the null process.
    fn matched_receive_vec<Msg: EquivalentDatatype>(self) -> (Option<Vec<Msg>>, Status);
}

impl MatchedReceiveVec for (Message, Status) {
    fn matched_receive_vec<Msg: EquivalentDatatype>(self) -> (Option<Vec<Msg>>, Status) {
        let is_no_proc = self.0.is_no_proc();
        let count = self.1.count(Msg::equivalent_datatype()).value_as().unwrap();
        let mut res = Vec::with_capacity(count);
        unsafe { res.set_len(count); }
        let status = self.0.matched_receive_into(&mut res[..]);
        if is_no_proc {
            (None, status)
        } else {
            (Some(res), status)
        }
    }
}

/// Receive a message containing a single instance of type `Msg`.
///
/// # Examples
///
/// ```no_run
/// use mpi::traits::*;
///
/// let universe = mpi::initialize().unwrap();
/// let world = universe.world();
///
/// let x = world.receive::<f64>();
/// ```
///
/// # Standard section(s)
///
/// 3.2.4
pub trait Receive {
    /// Receive a message from `Source` `&self` tagged `tag` containing a single instance of type
    /// `Msg` or `None` if receiving from the null process.
    fn receive_with_tag<Msg: EquivalentDatatype>(&self, tag: Tag) -> (Option<Msg>, Status);

    /// Receive a message from `Source` `&self` containing a single instance of type `Msg` or
    /// `None` if receiving from the null process.
    fn receive<Msg: EquivalentDatatype>(&self) -> (Option<Msg>, Status) {
        self.receive_with_tag(ffi::RSMPI_ANY_TAG)
    }
}

impl<Src: Source> Receive for Src {
    fn receive_with_tag<Msg: EquivalentDatatype>(&self, tag: Tag) -> (Option<Msg>, Status) {
        let mut res = unsafe { mem::uninitialized() };
        let status = self.receive_into_with_tag(&mut res, tag);
        if self.source_rank() == ffi::RSMPI_PROC_NULL {
            (None, status)
        } else {
            (Some(res), status)
        }
    }
}

/// Receive a message into a `Buffer`.
///
/// # Standard section(s)
///
/// 3.2.4
pub trait ReceiveInto {
    /// Receive a message from `Source` `&self` tagged `tag` into `Buffer` `buf`.
    ///
    /// Receiving from the null process leaves `buf` untouched.
    fn receive_into_with_tag<Buf: BufferMut + ?Sized>(&self, buf: &mut Buf, tag: Tag) -> Status;

    /// Receive a message from `Source` `&self` into `Buffer` `buf`.
    ///
    /// Receiving from the null process leaves `buf` untouched.
    fn receive_into<Buf: BufferMut + ?Sized>(&self, buf: &mut Buf) -> Status {
        self.receive_into_with_tag(buf, ffi::RSMPI_ANY_TAG)
    }
}

impl<Src: Source> ReceiveInto for Src {
    fn receive_into_with_tag<Buf: BufferMut + ?Sized>(&self, buf: &mut Buf, tag: Tag) -> Status {
        let mut status: MPI_Status = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Recv(buf.pointer_mut(), buf.count(), buf.datatype().raw(),
                self.source_rank(), tag, self.communicator().raw(), &mut status as *mut MPI_Status);
        }
        Status(status)
    }
}

/// Receive a message containing multiple instances of type `Msg` into a `Vec`.
///
/// # Standard section(s)
///
/// 3.2.4
pub trait ReceiveVec {
    /// Receive a message from `Source` `&self` tagged `tag` containing multiple instances of type
    /// `Msg` into a `Vec` or `None` if receiving from the null process.
    fn receive_vec_with_tag<Msg: EquivalentDatatype>(&self, tag: Tag) -> (Option<Vec<Msg>>, Status);

    /// Receive a message from `Source` `&self` containing multiple instances of type `Msg` into a
    /// `Vec` or `None` if receiving from the null process.
    ///
    /// # Examples
    /// See `examples/send_receive.rs`
    fn receive_vec<Msg: EquivalentDatatype>(&self) -> (Option<Vec<Msg>>, Status) {
        self.receive_vec_with_tag(ffi::RSMPI_ANY_TAG)
    }
}

impl<Src: Source> ReceiveVec for Src {
    fn receive_vec_with_tag<Msg: EquivalentDatatype>(&self, tag: Tag) -> (Option<Vec<Msg>>, Status) {
        self.matched_probe_with_tag(tag).matched_receive_vec()
    }
}

// TODO: rewrite these as free-standing functions taking `Destination` and `Source` and assert
// s.comm == d.comm?

/// Simultaneously send and receive single instances of types `S` and `R`.
///
/// # Standard section(s)
///
/// 3.10
pub trait SendReceive {
    /// Sends `msg` to `Rank` `destination` tagging it `sendtag` and simultaneously receives an
    /// instance of `R` tagged `receivetag` from `Rank` `source` or receives `None` if receiving
    /// from the null process.
    fn send_receive_with_tags<S, R>(&self,
                                    msg: &S,
                                    destination: Rank,
                                    sendtag: Tag,
                                    source: Rank,
                                    receivetag: Tag)
                                    -> (Option<R>, Status)
        where S: EquivalentDatatype,
              R: EquivalentDatatype;

    /// Sends `msg` to `Rank` `destination` and simultaneously receives an instance of `R` from
    /// `Rank` `source` or receives `None` if receiving from the null process.
    ///
    /// # Examples
    /// See `examples/send_receive.rs`
    fn send_receive<S, R>(&self,
                          msg: &S,
                          destination: Rank,
                          source: Rank)
                          -> (Option<R>, Status)
        where S: EquivalentDatatype,
              R: EquivalentDatatype
    {
        self.send_receive_with_tags(msg, destination, Tag::default(), source, ffi::RSMPI_ANY_TAG)
    }
}

impl<T: SendReceiveInto> SendReceive for T {
    fn send_receive_with_tags<S, R>(&self,
                                    msg: &S,
                                    destination: Rank,
                                    sendtag: Tag,
                                    source: Rank,
                                    receivetag: Tag)
                                    -> (Option<R>, Status)
        where S: EquivalentDatatype,
              R: EquivalentDatatype
    {
        let mut res = unsafe { mem::uninitialized() };
        let status = self.send_receive_into_with_tags(
            msg, destination, sendtag,
            &mut res, source, receivetag);
        if source == ffi::RSMPI_PROC_NULL {
            (None, status)
        } else {
            (Some(res), status)
        }
    }
}

/// Simultaneously send and receive the contents of buffers.
///
/// # Standard section(s)
///
/// 3.10
pub trait SendReceiveInto {
    /// Sends the contents of `sendbuf` to `Rank` `destination` tagging it `sendtag` and
    /// simultaneously receives a message tagged `receivetag` from `Rank` `source` into
    /// `receivebuf`.
    ///
    /// Receiving from the null process leaves `receivebuf` untouched.
    fn send_receive_into_with_tags<S: ?Sized, R: ?Sized>(&self,
                                                         sendbuf: &S,
                                                         destination: Rank,
                                                         sendtag: Tag,
                                                         receivebuf: &mut R,
                                                         source: Rank,
                                                         receivetag: Tag)
                                                         -> Status
        where S: Buffer,
              R: BufferMut;

    /// Sends the contents of `sendbuf` to `Rank` `destination` and
    /// simultaneously receives a message from `Rank` `source` into
    /// `receivebuf`.
    ///
    /// Receiving from the null process leaves `receivebuf` untouched.
    fn send_receive_into<S: ?Sized, R: ?Sized>(&self,
                                               sendbuf: &S,
                                               destination: Rank,
                                               receivebuf: &mut R,
                                               source: Rank)
                                               -> Status
        where S: Buffer,
              R: BufferMut
    {
        self.send_receive_into_with_tags(sendbuf, destination, Tag::default(), receivebuf, source, ffi::RSMPI_ANY_TAG)
    }
}

impl<C: RawCommunicator> SendReceiveInto for C {
    fn send_receive_into_with_tags<S: ?Sized, R: ?Sized>(&self,
                                                         sendbuf: &S,
                                                         destination: Rank,
                                                         sendtag: Tag,
                                                         receivebuf: &mut R,
                                                         source: Rank,
                                                         receivetag: Tag)
                                                         -> Status
        where S: Buffer,
              R: BufferMut
    {
        let mut status: MPI_Status = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Sendrecv(
                sendbuf.pointer(), sendbuf.count(), sendbuf.datatype().raw(), destination, sendtag,
                receivebuf.pointer_mut(), receivebuf.count(), receivebuf.datatype().raw(), source, receivetag,
                self.raw(), &mut status as *mut MPI_Status);
        }
        Status(status)
    }
}

/// Simultaneously send and receive the contents of the same buffer.
///
/// # Standard section(s)
///
/// 3.10
pub trait SendReceiveReplaceInto {
    /// Sends the contents of `buf` to `Rank` `destination` tagging it `sendtag` and
    /// simultaneously receives a message tagged `receivetag` from `Rank` `source` and replaces the
    /// contents of `buf` with it.
    ///
    /// Receiving from the null process leaves `buf` untouched.
    fn send_receive_replace_into_with_tags<B: ?Sized>(&self,
                                                      buf: &mut B,
                                                      destination: Rank,
                                                      sendtag: Tag,
                                                      source: Rank,
                                                      receivetag: Tag)
                                                      -> Status
        where B: BufferMut;

    /// Sends the contents of `buf` to `Rank` `destination` and
    /// simultaneously receives a message from `Rank` `source` into and replaces the contents of
    /// `buf` with it.
    ///
    /// Receiving from the null process leaves `buf` untouched.
    fn send_receive_replace_into<B: ?Sized>(&self,
                                            buf: &mut B,
                                            destination: Rank,
                                            source: Rank)
                                            -> Status
        where B: BufferMut
    {
        self.send_receive_replace_into_with_tags(buf, destination, Tag::default(), source, Tag::default())
    }
}

impl<C: RawCommunicator> SendReceiveReplaceInto for C {
    fn send_receive_replace_into_with_tags<B: ?Sized>(&self,
                                                      buf: &mut B,
                                                      destination: Rank,
                                                      sendtag: Tag,
                                                      source: Rank,
                                                      receivetag: Tag)
                                                      -> Status
        where B: BufferMut
    {
        let mut status: MPI_Status = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Sendrecv_replace(
                buf.pointer_mut(), buf.count(), buf.datatype().raw(), destination, sendtag,
                source, receivetag, self.raw(), &mut status as *mut MPI_Status);
        }
        Status(status)
    }
}

/// Something that can identify as a raw `MPI_Request`
pub trait RawRequest {
    /// The raw `MPI_Request` value
    unsafe fn raw(&self) -> MPI_Request;
    /// A mutable pointer to the raw `MPI_Request` value
    unsafe fn ptr_mut(&mut self) -> *mut MPI_Request;
}

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
            ffi::MPI_Wait(self.ptr_mut(), &mut status as *mut MPI_Status);
        }
        Status(status)
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
            ffi::MPI_Test(self.ptr_mut(), &mut flag as *mut c_int, &mut status as *mut MPI_Status);
        }
        if flag != 0 {
            Result::Ok(Status(status))
        } else {
            Result::Err(self)
        }
    }
}

impl<R: RawRequest + Sized> Test for R { }

/// A request object for an immediate (non-blocking) send operation
///
/// # Examples
///
/// See `examples/immediate.rs`
///
/// # Standard section(s)
///
/// 3.7.1
#[must_use]
pub struct SendRequest<'b, Buf: 'b + Buffer + ?Sized>(MPI_Request, PhantomData<&'b Buf>);

impl<'b, Buf: 'b + Buffer + ?Sized> RawRequest for SendRequest<'b, Buf> {
    unsafe fn raw(&self) -> MPI_Request { self.0 }
    unsafe fn ptr_mut(&mut self) -> *mut MPI_Request { &mut (self.0) }
}

impl<'b, Buf: 'b + Buffer + ?Sized> Drop for SendRequest<'b, Buf> {
    fn drop(&mut self) {
        unsafe {
            assert!(self.raw() == ffi::RSMPI_REQUEST_NULL,
                "asynchronous send request dropped without ascertaining completion.");
        }
    }
}

/// Initiate an immediate (non-blocking) standard mode send operation.
///
/// # Examples
/// See `examples/immediate.rs`
///
/// # Standard section(s)
///
/// 3.7.2
pub trait ImmediateSend {
    /// Initiate sending the data in `buf` in standard mode and tag it.
    fn immediate_send_with_tag<'b, Buf: 'b + Buffer + ?Sized>(&self, buf: &'b Buf, tag: Tag) -> SendRequest<'b, Buf>;

    /// Initiate sending the data in `buf` in standard mode.
    fn immediate_send<'b, Buf: 'b + Buffer + ?Sized>(&self, buf: &'b Buf) -> SendRequest<'b, Buf> {
        self.immediate_send_with_tag(buf, Tag::default())
    }
}

impl<Dest: Destination> ImmediateSend for Dest {
    fn immediate_send_with_tag<'b, Buf: 'b + Buffer + ?Sized>(&self, buf: &'b Buf, tag: Tag) -> SendRequest<'b, Buf> {
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Isend(buf.pointer(), buf.count(), buf.datatype().raw(),
                self.destination_rank(), tag, self.communicator().raw(),
                &mut request as *mut MPI_Request);
        }
        SendRequest(request, PhantomData)
    }
}

/// Initiate an immediate (non-blocking) buffered mode send operation.
///
/// # Standard section(s)
///
/// 3.7.2
pub trait ImmediateBufferedSend {
    /// Initiate sending the data in `buf` in buffered mode and tag it.
    fn immediate_buffered_send_with_tag<'b, Buf: 'b + Buffer + ?Sized>(&self, buf: &'b Buf, tag: Tag) -> SendRequest<'b, Buf>;

    /// Initiate sending the data in `buf` in buffered mode.
    fn immediate_buffered_send<'b, Buf: 'b + Buffer + ?Sized>(&self, buf: &'b Buf) -> SendRequest<'b, Buf> {
        self.immediate_buffered_send_with_tag(buf, Tag::default())
    }
}

impl<Dest: Destination> ImmediateBufferedSend for Dest {
    fn immediate_buffered_send_with_tag<'b, Buf: 'b + Buffer + ?Sized>(&self, buf: &'b Buf, tag: Tag) -> SendRequest<'b, Buf> {
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Ibsend(buf.pointer(), buf.count(), buf.datatype().raw(),
                self.destination_rank(), tag, self.communicator().raw(),
                &mut request as *mut MPI_Request);
        }
        SendRequest(request, PhantomData)
    }
}

/// Initiate an immediate (non-blocking) synchronous mode send operation.
///
/// # Standard section(s)
///
/// 3.7.2
pub trait ImmediateSynchronousSend {
    /// Initiate sending the data in `buf` in synchronous mode and tag it.
    fn immediate_synchronous_send_with_tag<'b, Buf: 'b + Buffer + ?Sized>(&self, buf: &'b Buf, tag: Tag) -> SendRequest<'b, Buf>;

    /// Initiate sending the data in `buf` in synchronous mode.
    fn immediate_synchronous_send<'b, Buf: 'b + Buffer + ?Sized>(&self, buf: &'b Buf) -> SendRequest<'b, Buf> {
        self.immediate_synchronous_send_with_tag(buf, Tag::default())
    }
}

impl<Dest: Destination> ImmediateSynchronousSend for Dest {
    fn immediate_synchronous_send_with_tag<'b, Buf: 'b + Buffer + ?Sized>(&self, buf: &'b Buf, tag: Tag) -> SendRequest<'b, Buf> {
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Issend(buf.pointer(), buf.count(), buf.datatype().raw(),
                self.destination_rank(), tag, self.communicator().raw(),
                &mut request as *mut MPI_Request);
        }
        SendRequest(request, PhantomData)
    }
}

/// Initiate an immediate (non-blocking) ready mode send operation.
///
/// # Examples
///
/// See `examples/immediate.rs`
///
/// # Standard section(s)
///
/// 3.7.2
pub trait ImmediateReadySend {
    /// Initiate sending the data in `buf` in ready mode and tag it.
    fn immediate_ready_send_with_tag<'b, Buf: 'b + Buffer + ?Sized>(&self, buf: &'b Buf, tag: Tag) -> SendRequest<'b, Buf>;

    /// Initiate sending the data in `buf` in ready mode.
    fn immediate_ready_send<'b, Buf: 'b + Buffer + ?Sized>(&self, buf: &'b Buf) -> SendRequest<'b, Buf> {
        self.immediate_ready_send_with_tag(buf, Tag::default())
    }
}

impl<Dest: Destination> ImmediateReadySend for Dest {
    fn immediate_ready_send_with_tag<'b, Buf: 'b + Buffer + ?Sized>(&self, buf: &'b Buf, tag: Tag) -> SendRequest<'b, Buf> {
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Irsend(buf.pointer(), buf.count(), buf.datatype().raw(),
                self.destination_rank(), tag, self.communicator().raw(),
                &mut request as *mut MPI_Request);
        }
        SendRequest(request, PhantomData)
    }
}

/// A request object for an immediate (non-blocking) receive operation
///
/// # Examples
///
/// See `examples/immediate.rs`
///
/// # Standard section(s)
///
/// 3.7.1
#[must_use]
pub struct ReceiveRequest<'b, Buf: 'b + BufferMut + ?Sized>(MPI_Request, PhantomData<&'b mut Buf>);

impl<'b, Buf: 'b + BufferMut + ?Sized> RawRequest for ReceiveRequest<'b, Buf> {
    unsafe fn raw(&self) -> MPI_Request { self.0 }
    unsafe fn ptr_mut(&mut self) -> *mut MPI_Request { &mut (self.0) }
}

impl<'b, Buf: 'b + BufferMut + ?Sized> Drop for ReceiveRequest<'b, Buf> {
    fn drop(&mut self) {
        unsafe {
            assert!(self.raw() == ffi::RSMPI_REQUEST_NULL,
                "asynchronous receive request dropped without ascertaining completion.");
        }
    }
}

/// Initiate an immediate (non-blocking) receive operation.
///
/// # Examples
/// See `examples/immediate.rs`
///
/// # Standard section(s)
///
/// 3.7.2
pub trait ImmediateReceiveInto {
    /// Initiate receiving a message matching `tag` into `buf`.
    fn immediate_receive_into_with_tag<'b, Buf: 'b + BufferMut + ?Sized>(&self, buf: &mut Buf, tag: Tag) -> ReceiveRequest<'b, Buf>;

    /// Initiate receiving a message into `buf`.
    fn immediate_receive_into<'b, Buf: 'b + BufferMut + ?Sized>(&self, buf: &'b mut Buf) -> ReceiveRequest<'b, Buf> {
        self.immediate_receive_into_with_tag(buf, ffi::RSMPI_ANY_TAG)
    }
}

impl<Src:Source> ImmediateReceiveInto for Src {
    fn immediate_receive_into_with_tag<'b, Buf: 'b + BufferMut + ?Sized>(&self, buf: &mut Buf, tag: Tag) -> ReceiveRequest<'b, Buf> {
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Irecv(buf.pointer_mut(), buf.count(), buf.datatype().raw(),
                self.source_rank(), tag, self.communicator().raw(),
                &mut request as *mut MPI_Request);
        }
        ReceiveRequest(request, PhantomData)
    }
}

/// Asynchronously probe a source for incoming messages.
///
/// Like `Probe` but returns a `None` immediately if there is no incoming message to be probed.
///
/// # Standard section(s)
///
/// 3.8.1
pub trait ImmediateProbe {
    /// Asynchronously probe `Source` `&self` for incoming messages with a certain tag.
    fn immediate_probe_with_tag(&self, tag: Tag) -> Option<Status>;

    /// Asynchronously probe `Source` `&self` for incoming messages with any tag.
    fn immediate_probe(&self) -> Option<Status> {
        self.immediate_probe_with_tag(ffi::RSMPI_ANY_TAG)
    }
}

impl<Src: Source> ImmediateProbe for Src {
    fn immediate_probe_with_tag(&self, tag: Tag) -> Option<Status> {
        let mut status: MPI_Status = unsafe { mem::uninitialized() };
        let mut flag: c_int = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Iprobe(self.source_rank(), tag, self.communicator().raw(),
                &mut flag as *mut c_int, &mut status as *mut MPI_Status);
        };
        if flag != 0 {
            Some(Status(status))
        } else {
            None
        }
    }
}

/// Asynchronously probe a source for incoming messages with guaranteed reception.
///
/// Like `MatchedProbe` but returns a `None` immediately if there is no incoming message to be
/// probed.
///
/// # Standard section(s)
///
/// 3.8.2
pub trait ImmediateMatchedProbe {
    /// Asynchronously probe `Source` `&self` for incoming messages with a certain tag.
    fn immediate_matched_probe_with_tag(&self, tag: Tag) -> Option<(Message, Status)>;

    /// Asynchronously probe `Source` `&self` for incoming messages with any tag.
    fn immediate_matched_probe(&self) -> Option<(Message, Status)> {
        self.immediate_matched_probe_with_tag(ffi::RSMPI_ANY_TAG)
    }
}

impl<Src: Source> ImmediateMatchedProbe for Src {
    fn immediate_matched_probe_with_tag(&self, tag: Tag) -> Option<(Message, Status)> {
        let mut message: MPI_Message = unsafe { mem::uninitialized() };
        let mut status: MPI_Status = unsafe { mem::uninitialized() };
        let mut flag: c_int = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Improbe(self.source_rank(), tag, self.communicator().raw(),
                &mut flag as *mut c_int, &mut message as *mut MPI_Message,
                &mut status as *mut MPI_Status);
        }
        if flag != 0 {
            Some((Message(message), Status(status)))
        } else {
            None
        }
    }
}

/// Asynchronously receive a previously probed message into a `Buffer`.
///
/// # Standard section(s)
///
/// 3.8.3
pub trait ImmediateMatchedReceiveInto {
    /// Asynchronously receive the message `&self` with contents matching `buf`.
    ///
    /// Receiving from the null process leaves `buf` untouched.
    fn immediate_matched_receive_into<'b, Buf: 'b + BufferMut + ?Sized>(self, buf: &mut Buf) -> ReceiveRequest<'b, Buf>;
}

impl ImmediateMatchedReceiveInto for Message {
    fn immediate_matched_receive_into<'b, Buf: 'b + BufferMut + ?Sized>(mut self, buf: &mut Buf) -> ReceiveRequest<'b, Buf> {
        let mut request: MPI_Request = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Imrecv(buf.pointer_mut(), buf.count(), buf.datatype().raw(),
                self.ptr_mut(), &mut request as *mut MPI_Request);
            assert_eq!(self.raw(), ffi::RSMPI_MESSAGE_NULL);
        }
        ReceiveRequest(request, PhantomData)
    }
}
