//! Organizing processes as groups and communicators
//!
//! Upon initialization of the library (via `initialize()` or `initialize_with_threading()`) a
//! singleton communication `Universe` is created. All parallel processes initially partaking in
//! the computation are organized in a context called the 'world communicator' which is available
//! as a property of the `Universe`. From the world communicator, other communicators can be
//! created. Processes can be addressed via their `Rank` within a specific communicator. This
//! information is encapsulated in an `Identifier`.
//!
//! # Unfinished features
//!
//! - **6.3**: Group management
//!   - **6.3.1**: Accessors, `MPI_Group_size()`, `MPI_Group_rank()`,
//!   `MPI_Group_translate_ranks()`
//!   - **6.3.2**: Constructors, `MPI_Group_union()`, `MPI_Group_intersection()`,
//!   `MPI_Group_difference()`, `MPI_Group_incl()`, `MPI_Group_excl()`, `MPI_Group_range_incl()`,
//!   `MPI_Group_range_excl()`
//! - **6.4**: Communicator management
//!   - **6.4.2**: Constructors, `MPI_Comm_dup_with_info()`, `MPI_Comm_idup()`, `MPI_Comm_creat()`,
//!   `MPI_Comm_create_group()`, `MPI_Comm_split_type()`
//!   - **6.4.4**: Info, `MPI_Comm_set_info()`, `MPI_Comm_get_info()`
//! - **6.6**: Inter-communication
//! - **6.7**: Caching
//! - **6.8**: Naming objects
//! - **7**: Process topologies
//! - **Parts of sections**: 8, 10, 12
use std::{mem, ptr};
use std::cmp::Ordering;
use std::marker::PhantomData;

use libc::c_int;

use ffi;
use ffi::{MPI_Comm, MPI_Group};

pub mod traits;

/// Something that can identify itself as a raw MPI communicator
pub trait RawCommunicator {
    unsafe fn raw(&self) -> MPI_Comm;
}

impl<'a, C: 'a + RawCommunicator> RawCommunicator for &'a C {
    unsafe fn raw(&self) -> MPI_Comm {
        (*self).raw()
    }
}

/// Something that has a communicator associated with it
pub trait Communicator {
    type Out: RawCommunicator;
    fn communicator(&self) -> Self::Out;
}

impl<'a, C: 'a + RawCommunicator> Communicator for &'a C {
    type Out = &'a C;
    fn communicator(&self) -> Self::Out {
        self
    }
}

/// Identifies a certain process within a communicator.
pub type Rank = c_int;

/// Global context
pub struct Universe(PhantomData<()>);

impl Universe {
    /// The 'world communicator'
    ///
    /// Contains all processes initially partaking in the computation.
    ///
    /// # Examples
    /// See `examples/simple.rs`
    pub fn world(&self) -> SystemCommunicator {
        SystemCommunicator(ffi::RSMPI_COMM_WORLD)
    }

    /// Level of multithreading supported by this MPI universe
    ///
    /// See the `Threading` enum.
    ///
    /// # Examples
    /// See `examples/init_with_threading.rs`
    pub fn threading_support(&self) -> Threading {
        let mut res: c_int = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Query_thread(&mut res as *mut c_int); }
        res.into()
    }
}

impl Drop for Universe {
    fn drop(&mut self) {
        unsafe { ffi::MPI_Finalize(); }
    }
}

/// Describes the various levels of multithreading that can be supported by an MPI library.
///
/// # Examples
/// See `examples/init_with_threading.rs`
///
/// # Standard section(s)
///
/// 12.4.3
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Threading {
    /// All processes partaking in the computation are single-threaded.
    Single,
    /// Processes may be multi-threaded, but MPI functions will only ever be called from the main
    /// thread.
    Funneled,
    /// Processes may be multi-threaded, but calls to MPI functions will not be made concurrently.
    /// The user is responsible for serializing the calls.
    Serialized,
    /// Processes may be multi-threaded with no restrictions on the use of MPI functions from the
    /// threads.
    Multiple,
}

impl Threading {
    fn raw(self) -> c_int {
        use self::Threading::*;
        match self {
            Single => ffi::RSMPI_THREAD_SINGLE,
            Funneled => ffi::RSMPI_THREAD_FUNNELED,
            Serialized => ffi::RSMPI_THREAD_SERIALIZED,
            Multiple => ffi::RSMPI_THREAD_MULTIPLE
        }
    }
}

impl PartialOrd<Threading> for Threading {
    fn partial_cmp(&self, other: &Threading) -> Option<Ordering> {
        self.raw().partial_cmp(&other.raw())
    }
}

impl Ord for Threading {
    fn cmp(&self, other: &Threading) -> Ordering {
        self.raw().cmp(&other.raw())
    }
}

impl From<c_int> for Threading {
    fn from(i: c_int) -> Threading {
        use self::Threading::*;
        if i == ffi::RSMPI_THREAD_SINGLE { return Single; }
        else if i == ffi::RSMPI_THREAD_FUNNELED { return Funneled; }
        else if i == ffi::RSMPI_THREAD_SERIALIZED { return Serialized; }
        else if i == ffi::RSMPI_THREAD_MULTIPLE { return Multiple; }
        panic!("Unknown threading level: {}", i)
    }
}

fn is_initialized() -> bool {
    let mut res: c_int = unsafe { mem::uninitialized() };
    unsafe { ffi::MPI_Initialized(&mut res as *mut c_int); }
    res != 0
}

/// Initialize MPI.
///
/// If the MPI library has not been initialized so far, initializes and returns a representation
/// of the MPI communication `Universe` which provides access to additional functions.
/// Otherwise returns `None`.
///
/// Equivalent to: `initialize_with_threading(Threading::Single)`
///
/// # Examples
/// See `examples/simple.rs`
///
/// # Standard section(s)
///
/// 8.7
pub fn initialize() -> Option<Universe> {
    initialize_with_threading(Threading::Single)
    .map(|x| x.0)
}

/// Initialize MPI with desired level of multithreading support.
///
/// If the MPI library has not been initialized so far, tries to initialize with the desired level
/// of multithreading support and returns the MPI communication `Universe` with access to
/// additional functions as well as the level of multithreading actually supported by the
/// implementation. Otherwise returns `None`.
///
/// # Examples
/// See `examples/init_with_threading.rs`
///
/// # Standard section(s)
///
/// 12.4.3
pub fn initialize_with_threading(threading: Threading) -> Option<(Universe, Threading)> {
    if is_initialized() {
        None
    } else {
        let mut provided: c_int = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Init_thread(ptr::null_mut(), ptr::null_mut(), threading.raw(),
                &mut provided as *mut c_int);
        }
        Some((Universe(PhantomData), provided.into()))
    }
}

/// A built-in communicator, e.g. `MPI_COMM_WORLD`
///
/// # Standard section(s)
///
/// 6.4
#[derive(Copy, Clone)]
pub struct SystemCommunicator(MPI_Comm);

impl RawCommunicator for SystemCommunicator {
    unsafe fn raw(&self) -> MPI_Comm {
        self.0
    }
}

impl Communicator for SystemCommunicator {
    type Out = SystemCommunicator;
    fn communicator(&self) -> Self::Out {
        *self
    }
}

/// A user-defined communicator
///
/// # Standard section(s)
///
/// 6.4
pub struct UserCommunicator(MPI_Comm);

impl RawCommunicator for UserCommunicator {
    unsafe fn raw(&self) -> MPI_Comm {
        self.0
    }
}

impl Drop for UserCommunicator {
    fn drop(&mut self) {
        unsafe { ffi::MPI_Comm_free(&mut self.0 as *mut MPI_Comm); }
        assert_eq!(self.0, ffi::RSMPI_COMM_NULL);
    }
}

/// A color used in a communicator split.
pub type Color = c_int;

/// A key used when determining the rank order of processes after a communicator split.
pub type Key = c_int;

/// Standard operations on communicators
pub trait CommunicatorExt: Sized + RawCommunicator {
    /// Number of processes in this communicator
    ///
    /// # Examples
    /// See `examples/simple.rs`
    ///
    /// # Standard section(s)
    ///
    /// 6.4.1
    fn size(&self) -> Rank {
        let mut res: Rank = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Comm_size(self.raw(), &mut res as *mut c_int); }
        res
    }

    /// The `Rank` that identifies the calling process within this communicator
    ///
    /// # Examples
    /// See `examples/simple.rs`
    ///
    /// # Standard section(s)
    ///
    /// 6.4.1
    fn rank(&self) -> Rank {
        let mut res: Rank = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Comm_rank(self.raw(), &mut res as *mut c_int); }
        res
    }

    /// Bundles a reference to this communicator with a specific `Rank` into an `Identifier`.
    ///
    /// # Examples
    /// See `examples/broadcast.rs` `examples/gather.rs` `examples/send_receive.rs`
    // FIXME: handle out-of-range ranks
    fn process_at_rank(&self, i: Rank) -> Identifier<Self> {
        Identifier { comm: self, rank: i }
    }

    /// An `Identifier` for the calling process
    fn this_process(&self) -> Identifier<Self> {
        let rank = self.rank();
        Identifier { comm: self, rank: rank }
    }

    /// Compare two communicators.
    ///
    /// See enum `CommunicatorRelation`.
    ///
    /// # Standard section(s)
    ///
    /// 6.4.1
    fn compare<C: RawCommunicator>(&self, other: &C) -> CommunicatorRelation {
        let mut res: c_int = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Comm_compare(self.raw(), other.raw(), &mut res as *mut c_int); }
        res.into()
    }

    /// Duplicate a communicator.
    ///
    /// # Examples
    ///
    /// See `examples/duplicate.rs`
    ///
    /// # Standard section(s)
    ///
    /// 6.4.2
    fn duplicate(&self) -> UserCommunicator {
        let mut newcomm: MPI_Comm = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Comm_dup(self.raw(), &mut newcomm as *mut MPI_Comm); }
        UserCommunicator(newcomm)
    }

    /// Split a communicator.
    ///
    /// Creates as many new communicators as distinct values of `color` are given. All processes
    /// with the same value of `color` join the same communicator.
    ///
    /// # Examples
    ///
    /// See `examples/split.rs`
    ///
    /// # Standard section(s)
    ///
    /// 6.4.2
    fn split<C: Into<Color>>(&self, color: C) -> UserCommunicator {
        self.split_with_key(color.into(), Key::default())
    }

    /// Split a communicator.
    ///
    /// Like `split()` but orders processes according to the value of `key` in the new
    /// communicators.
    ///
    /// # Standard section(s)
    ///
    /// 6.4.2
    fn split_with_key<C: Into<Color>, K: Into<Key>>(&self, color: C, key: K) -> UserCommunicator {
        let mut newcomm: MPI_Comm = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Comm_split(self.raw(), color.into(), key.into(),
            &mut newcomm as *mut MPI_Comm);
        }
        UserCommunicator(newcomm)
    }

    /// The group associated with this communicator
    ///
    /// # Standard section(s)
    ///
    /// 6.3.2
    fn group(&self) -> Group {
        let mut group = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Comm_group(self.raw(), &mut group as *mut MPI_Group); }
        Group(group)
    }
}

impl CommunicatorExt for SystemCommunicator { }

impl CommunicatorExt for UserCommunicator { }

/// The relation between two communicators.
///
/// # Standard section(s)
///
/// 6.4.1
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum CommunicatorRelation {
    /// Identical groups and same contexts
    Identical,
    /// Groups match in constituents and rank order, contexts differ
    Congruent,
    /// Group constituents match but rank order differs
    Similar,
    /// Otherwise
    Unequal,
}

impl From<c_int> for CommunicatorRelation {
    fn from(i: c_int) -> CommunicatorRelation {
        use self::CommunicatorRelation::*;
        // FIXME: Yuck! These should be made const.
        if i == ffi::RSMPI_IDENT { return Identical; }
        else if i == ffi::RSMPI_CONGRUENT { return Congruent; }
        else if i == ffi::RSMPI_SIMILAR { return Similar; }
        else if i == ffi::RSMPI_UNEQUAL { return Unequal; }
        panic!("Unknown communicator relation: {}", i)
    }
}

/// Identifies a process by its `Rank` within a certain communicator.
#[derive(Copy, Clone)]
pub struct Identifier<'a, C: 'a + RawCommunicator> {
    comm: &'a C,
    rank: Rank,
}

impl<'a, C: 'a + RawCommunicator> Identifier<'a, C> {
    /// The process rank
    pub fn rank(&self) -> Rank {
        self.rank
    }
}

impl<'a, C: 'a + RawCommunicator> Communicator for Identifier<'a, C> {
    type Out = &'a C;
    fn communicator(&self) -> Self::Out {
        self.comm
    }
}

/// A group of processes
///
/// # Standard section(s)
///
/// 6.2.1
pub struct Group(MPI_Group);

impl Group {
    pub unsafe fn raw(&self) -> MPI_Group {
        self.0
    }

    /// Compare two groups.
    ///
    /// # Standard section(s)
    ///
    /// 6.3.1
    pub fn compare(&self, other: &Group) -> GroupRelation {
        let mut relation: c_int = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Group_compare(self.raw(), other.raw(), &mut relation as *mut c_int); }
        relation.into()
    }
}

impl Drop for Group {
    fn drop(&mut self) {
        unsafe { ffi::MPI_Group_free(&mut self.0 as *mut MPI_Group); }
        assert_eq!(self.0, ffi::RSMPI_GROUP_NULL);
    }
}

/// The relation between two groups.
///
/// # Standard section(s)
///
/// 6.3.1
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum GroupRelation {
    /// Identical group members in identical order
    Identical,
    /// Identical group members in different order
    Similar,
    /// Otherwise
    Unequal,
}

impl From<c_int> for GroupRelation {
    fn from(i: c_int) -> GroupRelation {
        use self::GroupRelation::*;
        // FIXME: Yuck! These should be made const.
        if i == ffi::RSMPI_IDENT { return Identical; }
        else if i == ffi::RSMPI_SIMILAR { return Similar; }
        else if i == ffi::RSMPI_UNEQUAL { return Unequal; }
        panic!("Unknown group relation: {}", i)
    }
}
