//! Point to point communication
//!
//! Endpoints of communication are mostly described by types that implement the `Source` and
//! `Destination` trait. Communication operations are implemented as traits that have blanket
//! implementations for these two traits.
//!
//! # Unfinished features
//!
//! - **3.2.6**: `MPI_STATUS_IGNORE`
//! - **3.4**: Any communicator mode except standard:
//!   - buffered mode, `MPI_Bsend()`
//!   - synchronous mode, `MPI_Ssend()`
//!   - ready mode, `MPI_Rsend()`
//! - **3.6**: Buffer usage, `MPI_Buffer_attach()`, `MPI_Buffer_detach()`
//! - **3.7**: Nonblocking mode:
//!   - Sending, `MPI_Isend()`, `MPI_Ibsend()`, `MPI_Issend()`, `MPI_Irsend()`
//!   - Receiving, `MPI_Irecv()`
//!   - Completion, `MPI_Wait()`, `MPI_Test()`, `MPI_Request_free()`, `MPI_Waitany()`,
//!   `MPI_Waitall()`, `MPI_Waitsome()`, `MPI_Testany()`, `MPI_Testall()`, `MPI_Testsome()`,
//!   `MPI_Request_get_status()`
//! - **3.8**:
//!   - Nonblocking probe operations, `MPI_Iprobe()`, `MPI_Improbe()`, `MPI_Imrecv()`
//!   - Cancellation, `MPI_Cancel()`, `MPI_Test_cancelled()`
//! - **3.9**: Persistent requests, `MPI_Send_init()`, `MPI_Bsend_init()`, `MPI_Ssend_init()`,
//! `MPI_Rsend_init()`, `MPI_Recv_init()`, `MPI_Start()`, `MPI_Startall()`
//! - **3.10**: In-place send-receive operations, `MPI_Sendrecv_replace()`
//! - **3.11**: Null processes

use std::{mem, fmt};

use ffi;
use ffi::{MPI_Status, MPI_Message};
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
//        ffi::constants::MPI_ANY_SOURCE
//    }
//}

impl Source for SystemCommunicator {
    fn source_rank(&self) -> Rank {
        ffi::constants::MPI_ANY_SOURCE
    }
}

impl<'a> Source for &'a UserCommunicator {
    fn source_rank(&self) -> Rank {
        ffi::constants::MPI_ANY_SOURCE
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
            ffi::MPI_Send(buf.send_address(), buf.count(), buf.datatype().raw(),
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
        self.probe_with_tag(ffi::constants::MPI_ANY_TAG)
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
pub struct Message(MPI_Message);

impl Message {
    unsafe fn raw(&self) -> MPI_Message {
        self.0
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
        self.matched_probe_with_tag(ffi::constants::MPI_ANY_TAG)
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
    /// Receives the message `&self` which contains a single instance of type `Msg`.
    fn matched_receive<Msg: EquivalentDatatype>(&mut self) -> (Msg, Status);
}

impl MatchedReceive for Message {
    fn matched_receive<Msg: EquivalentDatatype>(&mut self) -> (Msg, Status) {
        let mut res: Msg = unsafe { mem::uninitialized() };
        let status = self.matched_receive_into(&mut res);
        (res, status)
    }
}

/// Receive a previously probed message into a `Buffer`.
///
/// # Standard section(s)
///
/// 3.8.3
pub trait MatchedReceiveInto {
    /// Receive the message `&self` with contents matching `buf`.
    fn matched_receive_into<Buf: Buffer + ?Sized>(&mut self, buf: &mut Buf) -> Status;
}

impl MatchedReceiveInto for Message {
    fn matched_receive_into<Buf: Buffer + ?Sized>(&mut self, buf: &mut Buf) -> Status {
        let mut status: MPI_Status = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Mrecv(buf.receive_address(), buf.count(), buf.datatype().raw(),
                &mut self.raw() as *mut MPI_Message, &mut status as *mut MPI_Status);
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
    /// Receives the message `&self` which contains multiple instances of type `Msg` into a `Vec`.
    fn matched_receive_vec<Msg: EquivalentDatatype>(&mut self) -> (Vec<Msg>, Status);
}

impl MatchedReceiveVec for (Message, Status) {
    fn matched_receive_vec<Msg: EquivalentDatatype>(&mut self) -> (Vec<Msg>, Status) {
        let count = self.1.count(Msg::equivalent_datatype()) as usize; // FIXME: this should be a checked cast.
        let mut res = Vec::with_capacity(count);
        unsafe { res.set_len(count); }
        let status = self.0.matched_receive_into(&mut res[..]);
        (res, status)
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
    /// `Msg`.
    fn receive_with_tag<Msg: EquivalentDatatype>(&self, tag: Tag) -> (Msg, Status);
    /// Receive a message from `Source` `&self` containing a single instance of type `Msg`.
    fn receive<Msg: EquivalentDatatype>(&self) -> (Msg, Status) {
        self.receive_with_tag(ffi::constants::MPI_ANY_TAG)
    }
}

impl<Src: ReceiveInto> Receive for Src {
    fn receive_with_tag<Msg: EquivalentDatatype>(&self, tag: Tag) -> (Msg, Status) {
        let mut res = unsafe { mem::uninitialized() };
        let status = self.receive_into_with_tag(&mut res, tag);
        (res, status)
    }
}

/// Receive a message into a `Buffer`.
///
/// # Standard section(s)
///
/// 3.2.4
pub trait ReceiveInto {
    /// Receive a message from `Source` `&self` tagged `tag` into `Buffer` `buf`.
    fn receive_into_with_tag<Buf: Buffer + ?Sized>(&self, buf: &mut Buf, tag: Tag) -> Status;
    /// Receive a message from `Source` `&self` into `Buffer` `buf`.
    fn receive_into<Buf: Buffer + ?Sized>(&self, buf: &mut Buf) -> Status {
        self.receive_into_with_tag(buf, ffi::constants::MPI_ANY_TAG)
    }
}

impl<Src: Source> ReceiveInto for Src {
    fn receive_into_with_tag<Buf: Buffer + ?Sized>(&self, buf: &mut Buf, tag: Tag) -> Status {
        let mut status: MPI_Status = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Recv(buf.receive_address(), buf.count(), buf.datatype().raw(),
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
    /// `Msg` into a `Vec`.
    fn receive_vec_with_tag<Msg: EquivalentDatatype>(&self, tag: Tag) -> (Vec<Msg>, Status);
    /// Receive a message from `Source` `&self` containing multiple instances of type `Msg` into a
    /// `Vec`.
    ///
    /// # Examples
    /// See `examples/send_receive.rs`
    fn receive_vec<Msg: EquivalentDatatype>(&self) -> (Vec<Msg>, Status) {
        self.receive_vec_with_tag(ffi::constants::MPI_ANY_TAG)
    }
}

impl<Src: Source> ReceiveVec for Src {
    fn receive_vec_with_tag<Msg: EquivalentDatatype>(&self, tag: Tag) -> (Vec<Msg>, Status) {
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
    /// instance of `R` tagged `receivetag` from `Rank` `source`.
    fn send_receive_with_tags<S, R>(&self,
                                    msg: &S,
                                    destination: Rank,
                                    sendtag: Tag,
                                    source: Rank,
                                    receivetag: Tag)
                                    -> (R, Status)
        where S: EquivalentDatatype,
              R: EquivalentDatatype;

    /// Sends `msg` to `Rank` `destination` and simultaneously receives an
    /// instance of `R` from `Rank` `source`.
    ///
    /// # Examples
    /// See `examples/send_receive.rs`
    fn send_receive<S, R>(&self,
                          msg: &S,
                          destination: Rank,
                          source: Rank)
                          -> (R, Status)
        where S: EquivalentDatatype,
              R: EquivalentDatatype
    {
        self.send_receive_with_tags(msg, destination, Tag::default(), source, ffi::constants::MPI_ANY_TAG)
    }
}

impl<T: SendReceiveInto> SendReceive for T {
    fn send_receive_with_tags<S, R>(&self,
                                    msg: &S,
                                    destination: Rank,
                                    sendtag: Tag,
                                    source: Rank,
                                    receivetag: Tag)
                                    -> (R, Status)
        where S: EquivalentDatatype,
              R: EquivalentDatatype
    {
        let mut res = unsafe { mem::uninitialized() };
        let status = self.send_receive_into_with_tags(
            msg, destination, sendtag,
            &mut res, source, receivetag);
        (res, status)
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
    fn send_receive_into_with_tags<S: ?Sized, R: ?Sized>(&self,
                                                         sendbuf: &S,
                                                         destination: Rank,
                                                         sendtag: Tag,
                                                         receivebuf: &mut R,
                                                         source: Rank,
                                                         receivetag: Tag)
                                                         -> Status
        where S: Buffer,
              R: Buffer;

    /// Sends the contents of `sendbuf` to `Rank` `destination` and
    /// simultaneously receives a message from `Rank` `source` into
    /// `receivebuf`.
    fn send_receive_into<S: ?Sized, R: ?Sized>(&self,
                                               sendbuf: &S,
                                               destination: Rank,
                                               receivebuf: &mut R,
                                               source: Rank)
                                               -> Status
        where S: Buffer,
              R: Buffer
    {
        self.send_receive_into_with_tags(sendbuf, destination, Tag::default(), receivebuf, source, ffi::constants::MPI_ANY_TAG)
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
              R: Buffer
    {
        let mut status: MPI_Status = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Sendrecv(
                sendbuf.send_address(), sendbuf.count(), sendbuf.datatype().raw(), destination, sendtag,
                receivebuf.receive_address(), receivebuf.count(), receivebuf.datatype().raw(), source, receivetag,
                self.raw(), &mut status as *mut MPI_Status);
        }
        Status(status)
    }
}
