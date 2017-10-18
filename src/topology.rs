//! Organizing processes as groups and communicators
//!
//! Processes are organized in communicators. All parallel processes initially partaking in
//! the computation are organized in a context called the 'world communicator' which is available
//! as a property of the `Universe`. From the world communicator, other communicators can be
//! created. Processes can be addressed via their `Rank` within a specific communicator. This
//! information is encapsulated in a `Process`.
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
use std::{mem, process};
use std::os::raw::{c_int, c_char};
use std::ffi::{CStr, CString};

use super::Tag;
use ffi;
use ffi::{MPI_Comm, MPI_Group};

use raw::traits::*;

use datatype::traits::*;

/// Topology traits
pub mod traits {
    pub use super::{AsCommunicator, Communicator, Group};
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

/// A built-in communicator, e.g. `MPI_COMM_WORLD`
///
/// # Standard section(s)
///
/// 6.4
#[derive(Copy, Clone)]
pub struct SystemCommunicator(MPI_Comm);

impl SystemCommunicator {
    /// The 'world communicator'
    ///
    /// Contains all processes initially partaking in the computation.
    ///
    /// # Examples
    /// See `examples/simple.rs`
    pub fn world() -> SystemCommunicator {
        SystemCommunicator::from_raw_unchecked(unsafe_extern_static!(ffi::RSMPI_COMM_WORLD))
    }

    /// If the raw value is the null handle returns `None`
    #[allow(dead_code)]
    fn from_raw(raw: MPI_Comm) -> Option<SystemCommunicator> {
        if raw == unsafe_extern_static!(ffi::RSMPI_COMM_NULL) {
            None
        } else {
            Some(SystemCommunicator(raw))
        }
    }

    /// Wraps the raw value without checking for null handle
    fn from_raw_unchecked(raw: MPI_Comm) -> SystemCommunicator {
        debug_assert_ne!(raw, unsafe_extern_static!(ffi::RSMPI_COMM_NULL));
        SystemCommunicator(raw)
    }
}

unsafe impl AsRaw for SystemCommunicator {
    type Raw = MPI_Comm;
    fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Communicator for SystemCommunicator {}

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
        if raw == unsafe_extern_static!(ffi::RSMPI_COMM_NULL) {
            None
        } else {
            Some(UserCommunicator(raw))
        }
    }

    /// Wraps the raw value without checking for null handle
    fn from_raw_unchecked(raw: MPI_Comm) -> UserCommunicator {
        debug_assert_ne!(raw, unsafe_extern_static!(ffi::RSMPI_COMM_NULL));
        UserCommunicator(raw)
    }
}

impl AsCommunicator for UserCommunicator {
    type Out = UserCommunicator;
    fn as_communicator(&self) -> &Self::Out {
        self
    }
}

unsafe impl AsRaw for UserCommunicator {
    type Raw = MPI_Comm;
    fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Communicator for UserCommunicator {}

impl Drop for UserCommunicator {
    fn drop(&mut self) {
        unsafe {
            ffi::MPI_Comm_free(&mut self.0);
        }
        assert_eq!(self.0, unsafe_extern_static!(ffi::RSMPI_COMM_NULL));
    }
}

/// A color used in a communicator split
#[derive(Copy, Clone, Debug)]
pub struct Color(c_int);

impl Color {
    /// Special color of undefined value
    pub fn undefined() -> Color {
        Color(unsafe_extern_static!(ffi::RSMPI_UNDEFINED))
    }

    /// A color of a certain value
    ///
    /// Valid values are non-negative.
    pub fn with_value(value: c_int) -> Color {
        if value < 0 {
            panic!("Value of color must be non-negative.")
        }
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
        unsafe {
            ffi::MPI_Comm_size(self.as_raw(), &mut res);
        }
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
        unsafe {
            ffi::MPI_Comm_rank(self.as_raw(), &mut res);
        }
        res
    }

    /// Bundles a reference to this communicator with a specific `Rank` into a `Process`.
    ///
    /// # Examples
    /// See `examples/broadcast.rs` `examples/gather.rs` `examples/send_receive.rs`
    fn process_at_rank(&self, r: Rank) -> Process<Self>
    where
        Self: Sized,
    {
        assert!(0 <= r && r < self.size());
        Process::by_rank_unchecked(self, r)
    }

    /// Returns an `AnyProcess` identifier that can be used, e.g. as a `Source` in point to point
    /// communication.
    fn any_process(&self) -> AnyProcess<Self>
    where
        Self: Sized,
    {
        AnyProcess(self)
    }

    /// A `Process` for the calling process
    fn this_process(&self) -> Process<Self>
    where
        Self: Sized,
    {
        let rank = self.rank();
        Process::by_rank_unchecked(self, rank)
    }

    /// Compare two communicators.
    ///
    /// See enum `CommunicatorRelation`.
    ///
    /// # Standard section(s)
    ///
    /// 6.4.1
    fn compare<C: ?Sized>(&self, other: &C) -> CommunicatorRelation
    where
        C: Communicator,
    {
        let mut res: c_int = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Comm_compare(self.as_raw(), other.as_raw(), &mut res);
        }
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
        unsafe {
            ffi::MPI_Comm_dup(self.as_raw(), &mut newcomm);
        }
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
        unsafe {
            ffi::MPI_Comm_split(self.as_raw(), color.as_raw(), key, &mut newcomm);
        }
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
    fn split_by_subgroup_collective<G: ?Sized>(&self, group: &G) -> Option<UserCommunicator>
    where
        G: Group,
    {
        let mut newcomm: MPI_Comm = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Comm_create(self.as_raw(), group.as_raw(), &mut newcomm);
        }
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
    fn split_by_subgroup<G: ?Sized>(&self, group: &G) -> Option<UserCommunicator>
    where
        G: Group,
    {
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
    fn split_by_subgroup_with_tag<G: ?Sized>(&self, group: &G, tag: Tag) -> Option<UserCommunicator>
    where
        G: Group,
    {
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
        unsafe {
            ffi::MPI_Comm_group(self.as_raw(), &mut group);
        }
        UserGroup(group)
    }

    /// Abort program execution
    ///
    /// # Standard section(s)
    ///
    /// 8.7
    fn abort(&self, errorcode: c_int) -> ! {
        unsafe {
            ffi::MPI_Abort(self.as_raw(), errorcode);
        }
        process::abort();
    }

    /// Set the communicator name
    ///
    /// # MPI standard reference
    /// Section 6.8, see the `MPI_Comm_set_name` function
    fn set_name(&self, name: &str) {
        let c_name = CString::new(name).expect("Failed to convert the Rust string to a C string");
        unsafe {
            ffi::MPI_Comm_set_name(self.as_raw(), c_name.as_ptr());
        }
    }

    /// Get the communicator name
    ///
    /// # MPI standard reference
    /// Section 6.8, see the `MPI_Comm_get_name` function
    fn get_name(&self) -> String {
        type BufType = [c_char; ffi::MPI_MAX_OBJECT_NAME as usize];

        unsafe {
            let mut buf: BufType = mem::uninitialized();
            let mut resultlen: c_int = mem::uninitialized();

            ffi::MPI_Comm_get_name(self.as_raw(), buf.as_mut_ptr(), &mut resultlen);

            let buf_cstr = CStr::from_ptr(buf.as_ptr());
            buf_cstr.to_string_lossy().into_owned()
        }
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
        if i == unsafe_extern_static!(ffi::RSMPI_IDENT) {
            return Identical;
        } else if i == unsafe_extern_static!(ffi::RSMPI_CONGRUENT) {
            return Congruent;
        } else if i == unsafe_extern_static!(ffi::RSMPI_SIMILAR) {
            return Similar;
        } else if i == unsafe_extern_static!(ffi::RSMPI_UNEQUAL) {
            return Unequal;
        }
        panic!("Unknown communicator relation: {}", i)
    }
}

/// Identifies a process by its `Rank` within a certain communicator.
#[derive(Copy, Clone)]
pub struct Process<'a, C>
where
    C: 'a + Communicator,
{
    comm: &'a C,
    rank: Rank,
}

impl<'a, C> Process<'a, C>
where
    C: 'a + Communicator,
{
    #[allow(dead_code)]
    fn by_rank(c: &'a C, r: Rank) -> Option<Self> {
        if r != unsafe_extern_static!(ffi::RSMPI_PROC_NULL) {
            Some(Process { comm: c, rank: r })
        } else {
            None
        }
    }

    fn by_rank_unchecked(c: &'a C, r: Rank) -> Self {
        Process { comm: c, rank: r }
    }

    /// The process rank
    pub fn rank(&self) -> Rank {
        self.rank
    }
}

impl<'a, C> AsCommunicator for Process<'a, C>
where
    C: 'a + Communicator,
{
    type Out = C;
    fn as_communicator(&self) -> &Self::Out {
        self.comm
    }
}

/// Identifies an arbitrary process that is a member of a certain communicator, e.g. for use as a
/// `Source` in point to point communication.
pub struct AnyProcess<'a, C>(&'a C)
where
    C: 'a + Communicator;

impl<'a, C> AsCommunicator for AnyProcess<'a, C>
where
    C: 'a + Communicator,
{
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
        SystemGroup(unsafe_extern_static!(ffi::RSMPI_GROUP_EMPTY))
    }
}

unsafe impl AsRaw for SystemGroup {
    type Raw = MPI_Group;
    fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Group for SystemGroup {}

/// A user-defined group of processes
///
/// # Standard section(s)
///
/// 6.2.1
pub struct UserGroup(MPI_Group);

impl Drop for UserGroup {
    fn drop(&mut self) {
        unsafe {
            ffi::MPI_Group_free(&mut self.0);
        }
        assert_eq!(self.0, unsafe_extern_static!(ffi::RSMPI_GROUP_NULL));
    }
}

unsafe impl AsRaw for UserGroup {
    type Raw = MPI_Group;
    fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Group for UserGroup {}

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
    fn union<G>(&self, other: &G) -> UserGroup
    where
        G: Group,
    {
        let mut newgroup: MPI_Group = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Group_union(self.as_raw(), other.as_raw(), &mut newgroup);
        }
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
    fn intersection<G>(&self, other: &G) -> UserGroup
    where
        G: Group,
    {
        let mut newgroup: MPI_Group = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Group_intersection(self.as_raw(), other.as_raw(), &mut newgroup);
        }
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
    fn difference<G>(&self, other: &G) -> UserGroup
    where
        G: Group,
    {
        let mut newgroup: MPI_Group = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Group_difference(self.as_raw(), other.as_raw(), &mut newgroup);
        }
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
        unsafe {
            ffi::MPI_Group_incl(self.as_raw(), ranks.count(), ranks.as_ptr(), &mut newgroup);
        }
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
        unsafe {
            ffi::MPI_Group_excl(self.as_raw(), ranks.count(), ranks.as_ptr(), &mut newgroup);
        }
        UserGroup(newgroup)
    }

    /// Number of processes in the group.
    ///
    /// # Standard section(s)
    ///
    /// 6.3.1
    fn size(&self) -> Rank {
        let mut res: Rank = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Group_size(self.as_raw(), &mut res);
        }
        res
    }

    /// Rank of this process within the group.
    ///
    /// # Standard section(s)
    ///
    /// 6.3.1
    fn rank(&self) -> Option<Rank> {
        let mut res: Rank = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Group_rank(self.as_raw(), &mut res);
        }
        if res == unsafe_extern_static!(ffi::RSMPI_UNDEFINED) {
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
    fn translate_rank<G>(&self, rank: Rank, other: &G) -> Option<Rank>
    where
        G: Group,
    {
        let mut res: Rank = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Group_translate_ranks(self.as_raw(), 1, &rank, other.as_raw(), &mut res);
        }
        if res == unsafe_extern_static!(ffi::RSMPI_UNDEFINED) {
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
    fn translate_ranks<G>(&self, ranks: &[Rank], other: &G) -> Vec<Option<Rank>>
    where
        G: Group,
    {
        ranks
            .iter()
            .map(|&r| self.translate_rank(r, other))
            .collect()
    }

    /// Compare two groups.
    ///
    /// # Standard section(s)
    ///
    /// 6.3.1
    fn compare<G>(&self, other: &G) -> GroupRelation
    where
        G: Group,
    {
        let mut relation: c_int = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Group_compare(self.as_raw(), other.as_raw(), &mut relation);
        }
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
        if i == unsafe_extern_static!(ffi::RSMPI_IDENT) {
            return Identical;
        } else if i == unsafe_extern_static!(ffi::RSMPI_SIMILAR) {
            return Similar;
        } else if i == unsafe_extern_static!(ffi::RSMPI_UNEQUAL) {
            return Unequal;
        }
        panic!("Unknown group relation: {}", i)
    }
}
