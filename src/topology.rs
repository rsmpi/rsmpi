//! Organizing processes as groups and communicators
//!
//! Upon initialization of the library (via `initialize()` or `initialize_with_threading()`) a
//! singleton communication `Universe` is created. All parallel processes initially partaking in
//! the computation are organized in a context called the 'world communicator' which is available
//! as a property of the `Universe`. From the world communicator, other communicators can be
//! created. Processes can be addressed via their `Rank` within a specific communicator. This
//! information is encapsulated in a `ProcessIdentifier`.
//!
//! # Unfinished features
//!
//! - **6.3**: Group management
//!   - **6.3.2**: Constructors, `MPI_Group_range_incl()`, `MPI_Group_range_excl()`
//! - **6.4**: Communicator management
//!   - **6.4.2**: Constructors, `MPI_Comm_dup_with_info()`, `MPI_Comm_idup()`,
//!     `MPI_Comm_split_type()`
//!   - **6.4.4**: Info, `MPI_Comm_set_info()`, `MPI_Comm_get_info()`
//! - **6.6**: Inter-communication
//! - **6.7**: Caching
//! - **6.8**: Naming objects
//! - **7**: Process topologies
//! - **Parts of sections**: 8, 10, 12
use std::{mem, ptr};
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::string::{FromUtf8Error};
use std::os::raw::{c_char, c_int, c_double};

use conv::ConvUtil;

use super::Tag;
use ffi;
use ffi::{MPI_Comm, MPI_Group};

use raw::traits::*;

use datatype::traits::*;

/// Topology traits
pub mod traits {
    pub use super::{
        Communicator,
        AsCommunicator,
        Group
    };
}

/// Something that has a communicator associated with it
pub trait AsCommunicator {
    /// The type of the associated communicator
    type Out: Communicator;
    /// Returns the associated communicator.
    fn as_communicator(&self) -> &Self::Out;
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
        SystemCommunicator::from_raw_unchecked(ffi::RSMPI_COMM_WORLD)
    }

    /// Level of multithreading supported by this MPI universe
    ///
    /// See the `Threading` enum.
    ///
    /// # Examples
    /// See `examples/init_with_threading.rs`
    pub fn threading_support(&self) -> Threading {
        let mut res: c_int = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Query_thread(&mut res); }
        res.into()
    }

    /// Names the processor that the calling process is running on.
    ///
    /// Can return an `Err` if the processor name is not a UTF-8 string.
    pub fn get_processor_name(&self) -> Result<String, FromUtf8Error> {
        let bufsize = ffi::RSMPI_MAX_PROCESSOR_NAME.value_as().expect(
            &format!("MPI_MAX_LIBRARY_SIZE ({}) cannot be expressed as a usize.",
                ffi::RSMPI_MAX_PROCESSOR_NAME)
            );
        let mut buf = vec![0u8; bufsize];
        let mut len: c_int = 0;

        unsafe { ffi::MPI_Get_processor_name(buf.as_mut_ptr() as *mut c_char, &mut len); }
        buf.truncate(len.value_as().expect(
            &format!("Length of processor name string ({}) cannot be expressed as a usize.", len)));
        String::from_utf8(buf)
    }

    /// Time in seconds since an arbitrary time in the past.
    ///
    /// The cheapest high-resolution timer available will be used.
    pub fn get_time(&self) -> c_double {
      unsafe { ffi::RSMPI_Wtime() }
    }

    /// Resolution of timer used in MPI_Wtime in seconds
    pub fn get_time_res(&self) -> c_double {
      unsafe { ffi::RSMPI_Wtick() }
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
    /// The raw value understood by the MPI C API
    fn as_raw(&self) -> c_int {
        use self::Threading::*;
        match *self {
            Single => ffi::RSMPI_THREAD_SINGLE,
            Funneled => ffi::RSMPI_THREAD_FUNNELED,
            Serialized => ffi::RSMPI_THREAD_SERIALIZED,
            Multiple => ffi::RSMPI_THREAD_MULTIPLE
        }
    }
}

impl PartialOrd<Threading> for Threading {
    fn partial_cmp(&self, other: &Threading) -> Option<Ordering> {
        self.as_raw().partial_cmp(&other.as_raw())
    }
}

impl Ord for Threading {
    fn cmp(&self, other: &Threading) -> Ordering {
        self.as_raw().cmp(&other.as_raw())
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

/// Whether the MPI library has been initialized
fn is_initialized() -> bool {
    let mut res: c_int = unsafe { mem::uninitialized() };
    unsafe { ffi::MPI_Initialized(&mut res); }
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
            ffi::MPI_Init_thread(ptr::null_mut(), ptr::null_mut(), threading.as_raw(),
                &mut provided);
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

impl SystemCommunicator {
    /// If the raw value is the null handle returns `None`
    #[allow(dead_code)]
    fn from_raw(raw: MPI_Comm) -> Option<SystemCommunicator> {
        if raw == ffi::RSMPI_COMM_NULL {
            None
        } else {
            Some(SystemCommunicator(raw))
        }
    }

    /// Wraps the raw value without checking for null handle
    fn from_raw_unchecked(raw: MPI_Comm) -> SystemCommunicator {
        debug_assert!(raw != ffi::RSMPI_COMM_NULL);
        SystemCommunicator(raw)
    }
}

impl AsRaw for SystemCommunicator {
    type Raw = MPI_Comm;
    unsafe fn as_raw(&self) -> Self::Raw { self.0 }
}

impl Communicator for SystemCommunicator { }

impl AsCommunicator for SystemCommunicator {
    type Out = SystemCommunicator;
    fn as_communicator(&self) -> &Self::Out {
        self
    }
}

/// A user-defined communicator
///
/// # Standard section(s)
///
/// 6.4
pub struct UserCommunicator(MPI_Comm);

impl UserCommunicator {
    /// If the raw value is the null handle returns `None`
    fn from_raw(raw: MPI_Comm) -> Option<UserCommunicator> {
        if raw == ffi::RSMPI_COMM_NULL {
            None
        } else {
            Some(UserCommunicator(raw))
        }
    }

    /// Wraps the raw value without checking for null handle
    fn from_raw_unchecked(raw: MPI_Comm) -> UserCommunicator {
        debug_assert!(raw != ffi::RSMPI_COMM_NULL);
        UserCommunicator(raw)
    }
}

impl AsCommunicator for UserCommunicator {
    type Out = UserCommunicator;
    fn as_communicator(&self) -> &Self::Out {
        self
    }
}

impl AsRaw for UserCommunicator {
    type Raw = MPI_Comm;
    unsafe fn as_raw(&self) -> Self::Raw { self.0 }
}

impl Communicator for UserCommunicator { }

impl Drop for UserCommunicator {
    fn drop(&mut self) {
        unsafe { ffi::MPI_Comm_free(&mut self.0); }
        assert_eq!(self.0, ffi::RSMPI_COMM_NULL);
    }
}

/// A color used in a communicator split
pub struct Color(c_int);

impl Color {
    /// Special color of undefined value
    pub fn undefined() -> Color {
        Color(ffi::RSMPI_UNDEFINED)
    }

    /// A color of a certain value
    ///
    /// Valid values are non-negative.
    pub fn with_value(value: c_int) -> Color {
        if value < 0 { panic!("Value of color must be non-negative.") }
        Color(value)
    }

    /// The raw value understood by the MPI C API
    fn as_raw(&self) -> c_int {
        self.0
    }
}

/// A key used when determining the rank order of processes after a communicator split.
pub type Key = c_int;

/// Communicators are contexts for communication
pub trait Communicator: AsRaw<Raw = MPI_Comm> {
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
        unsafe { ffi::MPI_Comm_size(self.as_raw(), &mut res); }
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
        unsafe { ffi::MPI_Comm_rank(self.as_raw(), &mut res); }
        res
    }

    /// Bundles a reference to this communicator with a specific `Rank` into a `ProcessIdentifier`.
    ///
    /// # Examples
    /// See `examples/broadcast.rs` `examples/gather.rs` `examples/send_receive.rs`
    fn process_at_rank(&self, r: Rank) -> ProcessIdentifier<Self> where Self: Sized {
        assert!(0 <= r && r < self.size());
        ProcessIdentifier { comm: self, rank: r }
    }

    /// Returns an `AnyProcess` identifier that can be used, e.g. as a `Source` in point to point
    /// communication.
    fn any_process(&self) -> AnyProcess<Self> where Self: Sized {
        AnyProcess(self)
    }

    /// The null process
    ///
    /// Point to point send/receive operations involving the null process as source/destination
    /// have no effect.
    ///
    /// # Examples
    /// See `examples/null_process.rs`
    ///
    /// # Standard section(s)
    ///
    /// 3.11
    fn null_process(&self) -> ProcessIdentifier<Self> where Self: Sized {
        ProcessIdentifier { comm: self, rank: ffi::RSMPI_PROC_NULL }
    }

    /// A `ProcessIdentifier` for the calling process
    fn this_process(&self) -> ProcessIdentifier<Self> where Self: Sized{
        let rank = self.rank();
        ProcessIdentifier { comm: self, rank: rank }
    }

    /// Compare two communicators.
    ///
    /// See enum `CommunicatorRelation`.
    ///
    /// # Standard section(s)
    ///
    /// 6.4.1
    fn compare<C: ?Sized + Communicator>(&self, other: &C) -> CommunicatorRelation {
        let mut res: c_int = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Comm_compare(self.as_raw(), other.as_raw(), &mut res); }
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
        unsafe { ffi::MPI_Comm_dup(self.as_raw(), &mut newcomm); }
        UserCommunicator::from_raw_unchecked(newcomm)
    }

    /// Split a communicator by color.
    ///
    /// Creates as many new communicators as distinct values of `color` are given. All processes
    /// with the same value of `color` join the same communicator. A process that passes the
    /// special undefined color will not join a new communicator and `None` is returned.
    ///
    /// # Examples
    ///
    /// See `examples/split.rs`
    ///
    /// # Standard section(s)
    ///
    /// 6.4.2
    fn split_by_color(&self, color: Color) -> Option<UserCommunicator> {
        self.split_by_color_with_key(color, Key::default())
    }

    /// Split a communicator by color.
    ///
    /// Like `split()` but orders processes according to the value of `key` in the new
    /// communicators.
    ///
    /// # Standard section(s)
    ///
    /// 6.4.2
    fn split_by_color_with_key(&self, color: Color, key: Key) -> Option<UserCommunicator> {
        let mut newcomm: MPI_Comm = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Comm_split(self.as_raw(), color.as_raw(), key, &mut newcomm); }
        UserCommunicator::from_raw(newcomm)
    }

    /// Split a communicator collectively by subgroup.
    ///
    /// Proceses pass in a group that is a subgroup of the group associated with the old
    /// communicator. Different processes may pass in different groups, but if two groups are
    /// different, they have to be disjunct. One new communicator is created for each distinct
    /// group. The new communicator is returned if a process is a member of the group he passed in,
    /// otherwise `None`.
    ///
    /// This call is a collective operation on the old communicator so all processes have to
    /// partake.
    ///
    /// # Examples
    ///
    /// See `examples/split.rs`
    ///
    /// # Standard section(s)
    ///
    /// 6.4.2
    fn split_by_subgroup_collective<G: ?Sized + Group>(&self, group: &G) -> Option<UserCommunicator> {
        let mut newcomm: MPI_Comm = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Comm_create(self.as_raw(), group.as_raw(), &mut newcomm); }
        UserCommunicator::from_raw(newcomm)
    }

    /// Split a communicator by subgroup.
    ///
    /// Like `split_by_subgroup_collective()` but not a collective operation.
    ///
    /// # Examples
    ///
    /// See `examples/split.rs`
    ///
    /// # Standard section(s)
    ///
    /// 6.4.2
    fn split_by_subgroup<G: ?Sized + Group>(&self, group: &G) -> Option<UserCommunicator> {
        self.split_by_subgroup_with_tag(group, Tag::default())
    }

    /// Split a communicator by subgroup
    ///
    /// Like `split_by_subgroup()` but can avoid collision of concurrent calls
    /// (i.e. multithreaded) by passing in distinct tags.
    ///
    /// # Standard section(s)
    ///
    /// 6.4.2
    fn split_by_subgroup_with_tag<G: ?Sized + Group>(&self, group: &G, tag: Tag) -> Option<UserCommunicator> {
        let mut newcomm: MPI_Comm = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Comm_create_group(self.as_raw(), group.as_raw(), tag, &mut newcomm);
        }
        UserCommunicator::from_raw(newcomm)
    }

    /// The group associated with this communicator
    ///
    /// # Standard section(s)
    ///
    /// 6.3.2
    fn group(&self) -> UserGroup {
        let mut group: MPI_Group = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Comm_group(self.as_raw(), &mut group); }
        UserGroup(group)
    }
}

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
pub struct ProcessIdentifier<'a, C: 'a + Communicator> {
    comm: &'a C,
    rank: Rank,
}

impl<'a, C: 'a + Communicator> ProcessIdentifier<'a, C> {
    /// The process rank
    pub fn rank(&self) -> Rank {
        self.rank
    }
}

impl<'a, C: 'a + Communicator> AsCommunicator for ProcessIdentifier<'a, C> {
    type Out = C;
    fn as_communicator(&self) -> &Self::Out {
        self.comm
    }
}

/// Identifies an arbitrary process that is a member of a certain communicator, e.g. for use as a
/// `Source` in point to point communication.
pub struct AnyProcess<'a, C: 'a + Communicator>(&'a C);

impl<'a, C: 'a + Communicator> AsCommunicator for AnyProcess<'a, C> {
    type Out = C;
    fn as_communicator(&self) -> &Self::Out {
        self.0
    }
}

/// A built-in group, e.g. `MPI_GROUP_EMPTY`
///
/// # Standard section(s)
///
/// 6.2.1
#[derive(Copy, Clone)]
pub struct SystemGroup(MPI_Group);

impl SystemGroup {
    /// An empty group
    pub fn empty() -> SystemGroup {
        SystemGroup(ffi::RSMPI_GROUP_EMPTY)
    }
}

impl AsRaw for SystemGroup {
    type Raw = MPI_Group;
    unsafe fn as_raw(&self) -> Self::Raw { self. 0 }
}

impl Group for SystemGroup { }

/// A user-defined group of processes
///
/// # Standard section(s)
///
/// 6.2.1
pub struct UserGroup(MPI_Group);

impl Drop for UserGroup {
    fn drop(&mut self) {
        unsafe { ffi::MPI_Group_free(&mut self.0); }
        assert_eq!(self.0, ffi::RSMPI_GROUP_NULL);
    }
}

impl AsRaw for UserGroup {
    type Raw = MPI_Group;
    unsafe fn as_raw(&self) -> Self::Raw { self.0 }
}

impl Group for UserGroup { }

/// Groups are collections of parallel processes
pub trait Group: AsRaw<Raw = MPI_Group> {
    /// Group union
    ///
    /// Constructs a new group that contains all members of the first group followed by all members
    /// of the second group that are not also members of the first group.
    ///
    /// # Standard section(s)
    ///
    /// 6.3.2
    fn union<G: Group>(&self, other: &G) -> UserGroup {
        let mut newgroup: MPI_Group = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Group_union(self.as_raw(), other.as_raw(), &mut newgroup); }
        UserGroup(newgroup)
    }

    /// Group intersection
    ///
    /// Constructs a new group that contains all processes that are members of both the first and
    /// second group in the order they have in the first group.
    ///
    /// # Standard section(s)
    ///
    /// 6.3.2
    fn intersection<G: Group>(&self, other: &G) -> UserGroup {
        let mut newgroup: MPI_Group = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Group_intersection(self.as_raw(), other.as_raw(), &mut newgroup); }
        UserGroup(newgroup)
    }

    /// Group difference
    ///
    /// Constructs a new group that contains all members of the first group that are not also
    /// members of the second group in the order they have in the first group.
    ///
    /// # Standard section(s)
    ///
    /// 6.3.2
    fn difference<G: Group>(&self, other: &G) -> UserGroup {
        let mut newgroup: MPI_Group = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Group_difference(self.as_raw(), other.as_raw(), &mut newgroup); }
        UserGroup(newgroup)
    }

    /// Subgroup including specified ranks
    ///
    /// Constructs a new group where the process with rank `ranks[i]` in the old group has rank `i`
    /// in the new group.
    ///
    /// # Standard section(s)
    ///
    /// 6.3.2
    fn include(&self, ranks: &[Rank]) -> UserGroup {
        let mut newgroup: MPI_Group = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Group_incl(self.as_raw(), ranks.count(), ranks.as_ptr(), &mut newgroup); }
        UserGroup(newgroup)
    }

    /// Subgroup including specified ranks
    ///
    /// Constructs a new group containing those processes from the old group that are not mentioned
    /// in `ranks`.
    ///
    /// # Standard section(s)
    ///
    /// 6.3.2
    fn exclude(&self, ranks: &[Rank]) -> UserGroup {
        let mut newgroup: MPI_Group = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Group_excl(self.as_raw(), ranks.count(), ranks.as_ptr(), &mut newgroup); }
        UserGroup(newgroup)
    }

    /// Number of processes in the group.
    ///
    /// # Standard section(s)
    ///
    /// 6.3.1
    fn size(&self) -> Rank {
        let mut res: Rank = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Group_size(self.as_raw(), &mut res); }
        res
    }

    /// Rank of this process within the group.
    ///
    /// # Standard section(s)
    ///
    /// 6.3.1
    fn rank(&self) -> Option<Rank> {
        let mut res: Rank = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Group_rank(self.as_raw(), &mut res); }
        if res == ffi::RSMPI_UNDEFINED {
            None
        } else {
            Some(res)
        }
    }

    /// Find the rank in group `other' of the process that has rank `rank` in this group.
    ///
    /// If the process is not a member of the other group, returns `None`.
    ///
    /// # Standard section(s)
    ///
    /// 6.3.1
    fn translate_rank<G: Group>(&self, rank: Rank, other: &G) -> Option<Rank> {
        let mut res: Rank = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Group_translate_ranks(self.as_raw(), 1, &rank, other.as_raw(), &mut res); }
        if res == ffi::RSMPI_UNDEFINED {
            None
        } else {
            Some(res)
        }
    }

    /// Find the ranks in group `other' of the processes that have ranks `ranks` in this group.
    ///
    /// If a process is not a member of the other group, returns `None`.
    ///
    /// # Standard section(s)
    ///
    /// 6.3.1
    fn translate_ranks<G: Group>(&self, ranks: &[Rank], other: &G) -> Vec<Option<Rank>> {
        ranks.iter().map(|&r| self.translate_rank(r, other)).collect()
    }

    /// Compare two groups.
    ///
    /// # Standard section(s)
    ///
    /// 6.3.1
    fn compare<G: Group>(&self, other: &G) -> GroupRelation {
        let mut relation: c_int = unsafe { mem::uninitialized() };
        unsafe { ffi::MPI_Group_compare(self.as_raw(), other.as_raw(), &mut relation); }
        relation.into()
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
