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
use std::ffi::{CStr, CString};
use std::mem::MaybeUninit;
use std::os::raw::{c_char, c_int, c_void};
use std::process;

use conv::ConvUtil;

#[cfg(not(msmpi))]
use crate::Tag;
use crate::{Count, IntArray};

use crate::datatype::traits::*;
use crate::ffi;
use crate::ffi::{MPI_Comm, MPI_Group};
use crate::raw::traits::*;
use crate::with_uninitialized;

mod cartesian;

/// Topology traits
pub mod traits {
    pub use super::{AsCommunicator, Attribute, Communicator, Group};
}

// Re-export cartesian functions and types from topology modules.
pub use self::cartesian::*;

/// Something that has a communicator associated with it
pub trait AsCommunicator {
    /// The type of the associated communicator
    type Out: Communicator;
    /// Returns the associated communicator.
    fn as_communicator(&self) -> &Self::Out;
}

/// Identifies a certain process within a communicator.
pub type Rank = c_int;

/// A raw communicator handle.
pub(crate) enum CommunicatorHandle {
    /// Built-in communicator `MPI_COMM_SELF`, containing only the current process. Exists until
    /// `MPI_Finalize` is called.
    ///
    /// # Standard section(s)
    ///
    /// 6.2
    SelfComm,

    /// Built-in communicator `MPI_COMM_WORLD`, containing all processes. Exists until
    /// `MPI_Finalize` is called.
    ///
    /// # Standard section(s)
    ///
    /// 6.2
    World,

    /// A user-defined communicator. Created through grouping operations such as `MPI_Comm_split`,
    /// must be freed when dropped.
    ///
    /// # Standard section(s)
    ///
    /// 6.4
    User(MPI_Comm),

    /// An inter-communicator returned by `MPI_Comm_get_parent`. Needs no drop semantics, because it
    /// was created in `MPI_Init`.
    ///
    /// # Standard section(s)
    ///
    /// 10.3
    Parent(MPI_Comm),

    /// A user-created inter-communicator that can needs to be disconnected when dropped.
    ///
    /// # Standard section(s)
    ///
    /// 6.6
    InterComm(MPI_Comm),
}

impl CommunicatorHandle {
    /// Create a `CommunicatorHandle` from a raw handle.
    ///
    /// # Returns
    /// * `None` if the handle is `MPI_COMM_NULL`
    /// * `SelfCommunicator` if the handle is `MPI_COMM_SELF`
    /// * `WorldCommunicator` if the handle is `MPI_COMM_WORLD`
    /// * `UserCommunicator` otherwise.
    ///
    /// # Safety
    /// Do not use for inter-communicator handles and inter-comm parent handles
    pub unsafe fn simple_from_raw(raw: MPI_Comm) -> Option<CommunicatorHandle> {
        if raw == ffi::RSMPI_COMM_NULL {
            None
        } else if raw == ffi::RSMPI_COMM_WORLD {
            Some(CommunicatorHandle::World)
        } else if raw == ffi::RSMPI_COMM_SELF {
            Some(CommunicatorHandle::SelfComm)
        } else {
            Some(CommunicatorHandle::User(raw))
        }
    }

    /// Create a `CommunicatorHandle::UserCommunicator` rom a raw handle without checking for
    /// null-handle, world-handle or self-handle
    ///
    /// # Safety
    /// Do not use for inter-communicator handles and inter-comm parent handles
    pub unsafe fn simple_from_raw_unchecked(raw: MPI_Comm) -> CommunicatorHandle {
        debug_assert_ne!(raw, ffi::RSMPI_COMM_NULL);
        debug_assert_ne!(raw, ffi::RSMPI_COMM_WORLD);
        debug_assert_ne!(raw, ffi::RSMPI_COMM_SELF);
        CommunicatorHandle::User(raw)
    }

    /// Create a `CommunicatorHandle::InterCommunicator` from a raw handle. Returns `None` if the
    /// handle is `MPI_COMM_NULL`.
    pub fn inter_from_raw(raw: MPI_Comm) -> Option<CommunicatorHandle> {
        if raw == unsafe { ffi::RSMPI_COMM_NULL } {
            None
        } else {
            let mut flag = c_int::min_value();
            unsafe {
                ffi::MPI_Comm_test_inter(raw, &mut flag);
            }

            let is_inter_comm = flag != 0;

            if is_inter_comm {
                Some(CommunicatorHandle::InterComm(raw))
            } else {
                None
            }
        }
    }

    /// Create a `CommunicatorHandle::InterCommunicator` from a raw handle without checking whether
    /// it is  `MPI_COMM_NULL` or an inter-comm.
    pub unsafe fn inter_from_raw_unchecked(raw: MPI_Comm) -> CommunicatorHandle {
        debug_assert_ne!(raw, ffi::RSMPI_COMM_NULL);
        debug_assert!({
            let mut flag = c_int::min_value();
            ffi::MPI_Comm_test_inter(raw, &mut flag);
            flag != 0
        });
        CommunicatorHandle::InterComm(raw)
    }

    /// Create a `CommunicatorHandle::ParentInterCommunicator` from a raw handle. Returns `None` if
    /// the handle is `MPI_COMM_NULL`.
    ///
    /// # Safety
    /// Do not call with handles that weren't obtained by `MPI_Comm_get_parent`
    pub unsafe fn parent_from_raw(raw: MPI_Comm) -> Option<CommunicatorHandle> {
        if raw == ffi::RSMPI_COMM_NULL {
            None
        } else {
            Some(CommunicatorHandle::Parent(raw))
        }
    }

    /// Returns true if the handle is of an inter-comm
    pub fn is_inter_comm(&self) -> bool {
        match self {
            CommunicatorHandle::SelfComm
            | CommunicatorHandle::World
            | CommunicatorHandle::User(_) => false,
            CommunicatorHandle::Parent(_) | CommunicatorHandle::InterComm(_) => true,
        }
    }
}

impl Drop for CommunicatorHandle {
    fn drop(&mut self) {
        match self {
            CommunicatorHandle::SelfComm => { /* cannot be dropped */ }
            CommunicatorHandle::World => { /* cannot be dropped */ }
            CommunicatorHandle::Parent(_) => { /* should be dropped */ }
            CommunicatorHandle::User(handle) => unsafe {
                ffi::MPI_Comm_free(handle);
                assert_eq!(*handle, ffi::RSMPI_COMM_NULL);
            },
            CommunicatorHandle::InterComm(handle) => unsafe {
                ffi::MPI_Comm_disconnect(handle);
                assert_eq!(*handle, ffi::RSMPI_COMM_NULL);
            },
        }
    }
}

unsafe impl AsRaw for CommunicatorHandle {
    type Raw = MPI_Comm;

    fn as_raw(&self) -> Self::Raw {
        match self {
            CommunicatorHandle::SelfComm => unsafe { ffi::RSMPI_COMM_SELF },
            CommunicatorHandle::World => unsafe { ffi::RSMPI_COMM_WORLD },
            CommunicatorHandle::Parent(handle) => *handle,
            CommunicatorHandle::User(handle) => *handle,
            CommunicatorHandle::InterComm(handle) => *handle,
        }
    }
}

/// A simple communicator, either a system-defined communicator like `MPI_COMM_WORLD` or a
/// user-defined intra-communicator without a special topology.
pub struct SimpleCommunicator(pub(crate) CommunicatorHandle);

impl SimpleCommunicator {
    /// The 'world communicator'
    ///
    /// Contains all processes initially partaking in the computation.
    ///
    /// # Examples
    /// See `examples/simple.rs`
    pub fn world() -> SimpleCommunicator {
        SimpleCommunicator(CommunicatorHandle::World)
    }

    /// The 'self communicator'
    ///
    /// Contains only the current process.
    pub fn self_comm() -> SimpleCommunicator {
        SimpleCommunicator(CommunicatorHandle::SelfComm)
    }

    /// If the raw value is the null handle returns `None`
    ///
    /// # Safety
    /// Do not call with inter-comm handles and inter-comm parent handles
    unsafe fn from_raw_checked(raw: MPI_Comm) -> Option<SimpleCommunicator> {
        let handle = CommunicatorHandle::simple_from_raw(raw)?;
        Some(SimpleCommunicator(handle))
    }

    /// Gets the topology of the communicator.
    ///
    /// # Standard section(s)
    /// 7.5.5
    pub fn topology(&self) -> Topology {
        unsafe {
            let (_, topology) =
                with_uninitialized(|topology| ffi::MPI_Topo_test(self.as_raw(), topology));

            if topology == ffi::RSMPI_GRAPH {
                Topology::Graph
            } else if topology == ffi::RSMPI_CART {
                Topology::Cartesian
            } else if topology == ffi::RSMPI_DIST_GRAPH {
                Topology::DistributedGraph
            } else if topology == ffi::RSMPI_UNDEFINED {
                Topology::Undefined
            } else {
                panic!("Unexpected Topology type!")
            }
        }
    }

    /// Converts the communicator into its precise communicator type.
    ///
    /// # Standard section(s)
    /// 7.5.5
    pub fn into_topology(self) -> IntoTopology {
        match self.topology() {
            Topology::Graph => unimplemented!(),
            Topology::Cartesian => IntoTopology::Cartesian(CartesianCommunicator(self)),
            Topology::DistributedGraph => unimplemented!(),
            Topology::Undefined => IntoTopology::Undefined(self),
        }
    }
}

unsafe impl AsRaw for SimpleCommunicator {
    type Raw = MPI_Comm;
    fn as_raw(&self) -> Self::Raw {
        self.0.as_raw()
    }
}

impl FromRaw for SimpleCommunicator {
    /// Wraps the raw value without checking for null handle
    ///
    /// # Safety
    /// - `handle` must be a live MPI_Comm object.
    /// - `handle` must not be an inter-comm handle, an inter-comm parent handle, `MPI_COMM_WORLD`,
    /// or `MPI_COMM_SELF`
    /// - `handle` must not be used after calling `from_raw`.
    unsafe fn from_raw(handle: <Self as AsRaw>::Raw) -> Self {
        debug_assert_ne!(handle, ffi::RSMPI_COMM_NULL);
        let handle = CommunicatorHandle::simple_from_raw_unchecked(handle);
        SimpleCommunicator(handle)
    }
}

impl Communicator for SimpleCommunicator {
    fn target_size(&self) -> Rank {
        self.size()
    }
}

impl AsCommunicator for SimpleCommunicator {
    type Out = SimpleCommunicator;
    fn as_communicator(&self) -> &Self::Out {
        self
    }
}

/// An enum describing the topology of a communicator
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Topology {
    /// Graph topology type
    Graph,
    /// Cartesian topology type
    Cartesian,
    /// DistributedGraph topology type
    DistributedGraph,
    /// Undefined topology type
    Undefined,
}

/// An enum indirecting between different concrete communicator topology types
pub enum IntoTopology {
    /// Graph topology type
    Graph(GraphCommunicator),
    /// Cartesian topology type
    Cartesian(CartesianCommunicator),
    /// DistributedGraph topology type
    DistributedGraph(DistributedGraphCommunicator),
    /// Undefined topology type
    Undefined(SimpleCommunicator),
}

/// A communicator for inter-communication, which represents a point-to-point communication between
/// disjoint groups
///
/// # Standard Sections
/// 6.6
pub struct InterCommunicator(pub(crate) CommunicatorHandle);

impl InterCommunicator {
    /// Construct an Intercommunicator from a handle
    pub(crate) fn from_handle(handle: CommunicatorHandle) -> Option<Self> {
        if handle.is_inter_comm() {
            Some(InterCommunicator(handle))
        } else {
            None
        }
    }

    /// Construct an Intercommunicator from a handle without checking if it's an Intercomm
    pub(crate) unsafe fn from_handle_unchecked(handle: CommunicatorHandle) -> Self {
        debug_assert!(handle.is_inter_comm());
        InterCommunicator(handle)
    }

    /// Construct an `InterCommunicator` from a raw handle
    pub fn from_raw_checked(raw: MPI_Comm) -> Option<Self> {
        CommunicatorHandle::inter_from_raw(raw)
            .map(|handle| unsafe { Self::from_handle_unchecked(handle) })
    }

    /// The number of processes in the remote group of comm
    ///
    /// # Standard Section(s)
    ///
    /// 6.6.1, See MPI_Comm_remote_size
    pub fn remote_size(&self) -> Rank {
        let mut size = Rank::min_value();
        unsafe {
            ffi::MPI_Comm_remote_size(self.as_raw(), &mut size);
        }
        size
    }

    /// The remote group corresponding to the intercomm
    ///
    /// # Standard Section(s)
    /// 6.6.1, See MPI_Comm_remote_group
    pub fn remote_group(&self) -> UserGroup {
        unsafe {
            let (_, g) = with_uninitialized(|g| {
                ffi::MPI_Comm_remote_group(self.as_raw(), g);
            });
            UserGroup(g)
        }
    }

    /// Merge an InterCommunicator into a single UserCommunicator
    ///
    /// To specify the ordering of merged ranks, every member of the one group
    /// should call with `MergeOrder::Low` and every member of the other group
    /// should use `MergeOrder::High`.
    ///
    /// # Standard section(s)
    ///
    /// 7.6.2
    pub fn merge(&self, merge_order: MergeOrder) -> SimpleCommunicator {
        unsafe {
            SimpleCommunicator::from_raw_checked(
                with_uninitialized(|raw| {
                    ffi::MPI_Intercomm_merge(self.as_raw(), merge_order.as_raw(), raw)
                })
                    .1,
            )
        }.expect("rspmi internal error: MPI implementation return MPI_COMM_NULL from MPI_Intercomm_merge()")
    }
}

impl AsCommunicator for InterCommunicator {
    type Out = InterCommunicator;
    fn as_communicator(&self) -> &Self::Out {
        self
    }
}

unsafe impl AsRaw for InterCommunicator {
    type Raw = MPI_Comm;
    fn as_raw(&self) -> Self::Raw {
        self.0.as_raw()
    }
}

impl FromRaw for InterCommunicator {
    /// Construct an `InterCommunicator` from a raw handle without checking if it's an Intercomm
    /// handle
    /// # Safety
    /// - `handle` must be a live MPI_Comm object.
    /// - `handle` must be an inter-comms or inter-comm parent handle
    /// - `handle` must not be used after calling `from_raw`.
    unsafe fn from_raw(handle: <Self as AsRaw>::Raw) -> Self {
        Self::from_handle_unchecked(CommunicatorHandle::inter_from_raw_unchecked(handle))
    }
}

impl Communicator for InterCommunicator {
    fn target_size(&self) -> Rank {
        self.remote_size()
    }
}

/// Unimplemented
#[allow(missing_copy_implementations)]
pub struct GraphCommunicator;

/// Unimplemented
#[allow(missing_copy_implementations)]
pub struct DistributedGraphCommunicator;

/// A color used in a communicator split
#[derive(Copy, Clone, Debug)]
pub struct Color(c_int);

impl Color {
    /// Special color of undefined value
    pub fn undefined() -> Color {
        Color(unsafe { ffi::RSMPI_UNDEFINED })
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
    fn as_raw(self) -> c_int {
        self.0
    }
}

/// A key used when determining the rank order of processes after a communicator split.
pub type Key = c_int;

/// Communicators are contexts for communication
pub trait Communicator: AsRaw<Raw = MPI_Comm> {
    /// Returns the number of processes available to communicate with in this `Communicator`. For
    /// intra-communicators, this is equivalent to [`size`](#method.size). For inter-communicators,
    /// this is equivalent to [`remote_size`](struct.InterCommunicator.html#method.remote_size).
    fn target_size(&self) -> Rank;

    /// Number of processes in this communicator
    ///
    /// # Examples
    /// See `examples/simple.rs`
    ///
    /// # Standard section(s)
    ///
    /// 6.4.1
    fn size(&self) -> Rank {
        unsafe { with_uninitialized(|size| ffi::MPI_Comm_size(self.as_raw(), size)).1 }
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
        unsafe { with_uninitialized(|rank| ffi::MPI_Comm_rank(self.as_raw(), rank)).1 }
    }

    /// Get `Attribute`
    fn get_attr<A: Attribute>(&self, key: A) -> Option<&<A as Attribute>::Target>
    where
        Self: Sized,
    {
        let (val, flag) = unsafe {
            let mut ptr: MaybeUninit<*mut <A as Attribute>::Target> = MaybeUninit::uninit();
            let (_, flag) = with_uninitialized(|flag| {
                ffi::MPI_Comm_get_attr(
                    self.as_raw(),
                    key.as_raw(),
                    ptr.as_mut_ptr() as *mut c_void,
                    flag,
                )
            });
            (ptr.assume_init(), flag)
        };
        if flag == 0 {
            None
        } else {
            unsafe { val.as_ref() }
        }
    }

    /// Bundles a reference to this communicator with a specific `Rank` into a `Process`.
    ///
    /// # Examples
    /// See `examples/broadcast.rs` `examples/gather.rs` `examples/send_receive.rs`
    fn process_at_rank(&self, r: Rank) -> Process<Self>
    where
        Self: Sized,
    {
        assert!(0 <= r && r < self.target_size());
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
        Self: Sized,
    {
        unsafe {
            with_uninitialized(|cmp| ffi::MPI_Comm_compare(self.as_raw(), other.as_raw(), cmp))
                .1
                .into()
        }
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
    fn duplicate(&self) -> SimpleCommunicator {
        unsafe {
            SimpleCommunicator::from_raw(
                with_uninitialized(|newcomm| ffi::MPI_Comm_dup(self.as_raw(), newcomm)).1,
            )
        }
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
    fn split_by_color(&self, color: Color) -> Option<SimpleCommunicator> {
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
    fn split_by_color_with_key(&self, color: Color, key: Key) -> Option<SimpleCommunicator> {
        unsafe {
            SimpleCommunicator::from_raw_checked(
                with_uninitialized(|newcomm| {
                    ffi::MPI_Comm_split(self.as_raw(), color.as_raw(), key, newcomm)
                })
                .1,
            )
        }
    }

    /// Split the communicator into subcommunicators, each of which can create a shared memory
    /// region.
    ///
    /// Within each subgroup, the processes are ranked in the order defined by the value of the
    /// argument key, with ties broken according to their rank in the old group.
    ///
    /// # Standard section(s)
    ///
    /// 6.4.2 (See: `MPI_Comm_split_type`)
    fn split_shared(&self, key: c_int) -> SimpleCommunicator {
        unsafe {
            SimpleCommunicator::from_raw_checked(
                with_uninitialized(|newcomm| {
                    ffi::MPI_Comm_split_type(
                        self.as_raw(),
                        ffi::RSMPI_COMM_TYPE_SHARED,
                        key,
                        ffi::RSMPI_INFO_NULL,
                        newcomm,
                    )
                })
                    .1,
            ).expect("rsmpi internal error: MPI implementation incorrectly returned MPI_COMM_NULL from MPI_Comm_split_type(..., MPI_COMM_TYPE_SHARED, ...)")
        }
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
    fn split_by_subgroup_collective<G: ?Sized>(&self, group: &G) -> Option<SimpleCommunicator>
    where
        G: Group,
        Self: Sized,
    {
        unsafe {
            SimpleCommunicator::from_raw_checked(
                with_uninitialized(|newcomm| {
                    ffi::MPI_Comm_create(self.as_raw(), group.as_raw(), newcomm)
                })
                .1,
            )
        }
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
    #[cfg(not(msmpi))]
    fn split_by_subgroup<G: ?Sized>(&self, group: &G) -> Option<SimpleCommunicator>
    where
        G: Group,
        Self: Sized,
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
    #[cfg(not(msmpi))]
    fn split_by_subgroup_with_tag<G: ?Sized>(
        &self,
        group: &G,
        tag: Tag,
    ) -> Option<SimpleCommunicator>
    where
        G: Group,
        Self: Sized,
    {
        unsafe {
            SimpleCommunicator::from_raw_checked(
                with_uninitialized(|newcomm| {
                    ffi::MPI_Comm_create_group(self.as_raw(), group.as_raw(), tag, newcomm)
                })
                .1,
            )
        }
    }

    /// The group associated with this communicator
    ///
    /// # Standard section(s)
    ///
    /// 6.3.2
    fn group(&self) -> UserGroup {
        unsafe {
            UserGroup(with_uninitialized(|group| ffi::MPI_Comm_group(self.as_raw(), group)).1)
        }
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

    /// Tests if the communicator is an inter-communicator.
    ///
    /// # Standard sections(s)
    ///
    /// 6.6.1, See MPI_Comm_test_inter
    fn test_inter(&self) -> bool {
        let mut flag = c_int::min_value();
        unsafe {
            ffi::MPI_Comm_test_inter(self.as_raw(), &mut flag);
        }
        flag != 0
    }

    /// Set the communicator name
    ///
    /// # Standard section(s)
    ///
    /// 6.8, see the `MPI_Comm_set_name` function
    fn set_name(&self, name: &str) {
        let c_name = CString::new(name).expect("Failed to convert the Rust string to a C string");
        unsafe {
            ffi::MPI_Comm_set_name(self.as_raw(), c_name.as_ptr());
        }
    }

    /// Get the communicator name
    ///
    /// # Standard section(s)
    ///
    /// 6.8, see the `MPI_Comm_get_name` function
    fn get_name(&self) -> String {
        type BufType = [c_char; ffi::MPI_MAX_OBJECT_NAME as usize];

        unsafe {
            let mut buf = MaybeUninit::<BufType>::uninit();

            let (_, _resultlen) = with_uninitialized(|resultlen| {
                ffi::MPI_Comm_get_name(self.as_raw(), &mut (*buf.as_mut_ptr())[0], resultlen)
            });

            let buf_cstr = CStr::from_ptr(buf.assume_init().as_ptr());
            buf_cstr.to_string_lossy().into_owned()
        }
    }

    /// Creates a communicator with ranks laid out in a multi-dimensional space, allowing for easy
    /// neighbor-to-neighbor communication, while providing MPI with information to allow it to
    /// better optimize the physical locality of ranks that are logically close.
    ///
    /// * `dims` - array of spatial extents for the cartesian space
    /// * `periods` - Must match length of `dims`. For `i` in 0 to `dims.len()`, `periods[i]` indicates if
    ///     axis `i` is periodic. i.e. if `true`, the element at `dims[i] - 1` in axis `i` is a neighbor of
    ///     element 0 in axis `i`
    /// * `reorder` - If true, MPI may re-order ranks in the new communicator.
    ///
    /// # Standard section(s)
    /// 7.5.1 (MPI_Cart_create)
    fn create_cartesian_communicator(
        &self,
        dims: &[Count],
        periods: &[bool],
        reorder: bool,
    ) -> Option<CartesianCommunicator> {
        assert_eq!(
            dims.len(),
            periods.len(),
            "dims and periods must be parallel, equal-sized arrays"
        );

        let periods: IntArray = periods.iter().map(|x| *x as i32).collect();

        unsafe {
            let mut comm_cart = ffi::RSMPI_COMM_NULL;
            ffi::MPI_Cart_create(
                self.as_raw(),
                dims.count(),
                dims.as_ptr(),
                periods.as_ptr(),
                reorder as Count,
                &mut comm_cart,
            );
            CartesianCommunicator::from_raw_checked(comm_cart)
        }
    }

    /// Gets the target rank of this rank as-if
    /// [`create_cartesian_communicator`](#method.create_cartesian_communicator) had been called
    /// with `dims`, `periods`, and `reorder = true`.
    ///
    /// Returns `None` if the local process would not particate in the new CartesianCommunciator.
    ///
    /// * `dims` - array of spatial extents for the cartesian space
    /// * `periods` - Must match length of `dims`. For `i` in 0 to `dims.len()`, `periods[i]` indicates if
    ///     axis `i` is periodic. i.e. if `true`, the element at `dims[i] - 1` in axis `i` is a neighbor of
    ///     element 0 in axis `i`
    ///
    /// # Standard section
    /// 7.5.8 (MPI_Cart_map)
    fn cartesian_map(&self, dims: &[Count], periods: &[bool]) -> Option<Rank> {
        assert_eq!(
            dims.len(),
            periods.len(),
            "dims and periods must be parallel, equal-sized arrays"
        );

        let periods: IntArray = periods.iter().map(|x| *x as i32).collect();

        unsafe {
            let mut new_rank = ffi::MPI_UNDEFINED;
            ffi::MPI_Cart_map(
                self.as_raw(),
                dims.count(),
                dims.as_ptr(),
                periods.as_ptr(),
                &mut new_rank,
            );
            if new_rank == ffi::MPI_UNDEFINED {
                None
            } else {
                Some(new_rank)
            }
        }
    }

    /// Gets the implementation-defined buffer size required to pack 'incount' elements of type
    /// 'datatype'.
    ///
    /// # Standard section(s)
    ///
    /// 4.2, see MPI_Pack_size
    fn pack_size<Dt>(&self, incount: Count, datatype: &Dt) -> Count
    where
        Dt: Datatype,
        Self: Sized,
    {
        unsafe {
            with_uninitialized(|size| {
                ffi::MPI_Pack_size(incount, datatype.as_raw(), self.as_raw(), size)
            })
            .1
        }
    }

    /// Packs inbuf into a byte array with an implementation-defined format. Often paired with
    /// `unpack` to convert back into a specific datatype.
    ///
    /// # Standard Sections
    ///
    /// 4.2, see MPI_Pack
    fn pack<Buf>(&self, inbuf: &Buf) -> Vec<u8>
    where
        Buf: ?Sized + Buffer,
        Self: Sized,
    {
        let inbuf_dt = inbuf.as_datatype();

        let mut outbuf = vec![
            0;
            self.pack_size(inbuf.count(), &inbuf_dt)
                .value_as::<usize>()
                .expect("MPI_Pack_size returned a negative buffer size!")
        ];

        let position = self.pack_into(inbuf, &mut outbuf[..], 0);

        outbuf.resize(
            position
                .value_as()
                .expect("MPI_Pack returned a negative position!"),
            0,
        );

        outbuf
    }

    /// Packs inbuf into a byte array with an implementation-defined format. Often paired with
    /// `unpack` to convert back into a specific datatype.
    ///
    /// # Standard Sections
    ///
    /// 4.2, see MPI_Pack
    fn pack_into<Buf>(&self, inbuf: &Buf, outbuf: &mut [u8], position: Count) -> Count
    where
        Buf: ?Sized + Buffer,
        Self: Sized,
    {
        let inbuf_dt = inbuf.as_datatype();

        let mut position: Count = position;
        unsafe {
            ffi::MPI_Pack(
                inbuf.pointer(),
                inbuf.count(),
                inbuf_dt.as_raw(),
                outbuf.as_mut_ptr() as *mut _,
                outbuf.count(),
                &mut position,
                self.as_raw(),
            );
        }
        position
    }

    /// Unpacks an implementation-specific byte array from `pack` or `pack_into` into a buffer of a
    /// specific datatype.
    ///
    /// # Standard Sections
    ///
    /// 4.2, see MPI_Unpack
    unsafe fn unpack_into<Buf>(&self, inbuf: &[u8], outbuf: &mut Buf, position: Count) -> Count
    where
        Buf: ?Sized + BufferMut,
        Self: Sized,
    {
        let outbuf_dt = outbuf.as_datatype();

        let mut position: Count = position;
        ffi::MPI_Unpack(
            inbuf.as_ptr() as *const _,
            inbuf.count(),
            &mut position,
            outbuf.pointer_mut(),
            outbuf.count(),
            outbuf_dt.as_raw(),
            self.as_raw(),
        );
        position
    }

    /// Returns the parent Communicator, if any
    ///
    /// # Standard Sections
    /// 10.3.2, see MPI_Comm_get_parent
    fn parent(&self) -> Option<InterCommunicator> {
        let handle = unsafe {
            let mut comm = ffi::RSMPI_COMM_NULL;
            ffi::MPI_Comm_get_parent(&mut comm);
            CommunicatorHandle::parent_from_raw(comm)?
        };
        InterCommunicator::from_handle(handle)
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
        if i == unsafe { ffi::RSMPI_IDENT } {
            return CommunicatorRelation::Identical;
        } else if i == unsafe { ffi::RSMPI_CONGRUENT } {
            return CommunicatorRelation::Congruent;
        } else if i == unsafe { ffi::RSMPI_SIMILAR } {
            return CommunicatorRelation::Similar;
        } else if i == unsafe { ffi::RSMPI_UNEQUAL } {
            return CommunicatorRelation::Unequal;
        }
        panic!("Unknown communicator relation: {}", i)
    }
}

/// When an intercommunicator is merged, the caller chooses how to order the two
/// groups. If every rank in one group uses `Low` and every rank in the other
/// use `High`, then they will be ordered accordingly. If both groups use the
/// same value, the order in the merged communicator is arbitrary.
#[derive(Copy, Clone)]
pub enum MergeOrder {
    /// Ranks in this group will be ordered first in the merged communicator
    Low,
    /// Ranks in this group will be ordered last in the merged communicator
    High,
}

impl MergeOrder {
    fn as_raw(&self) -> c_int {
        match self {
            MergeOrder::Low => 0,
            MergeOrder::High => 1,
        }
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
        if r != unsafe { ffi::RSMPI_PROC_NULL } {
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
        SystemGroup(unsafe { ffi::RSMPI_GROUP_EMPTY })
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
        assert_eq!(self.0, unsafe { ffi::RSMPI_GROUP_NULL });
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
        Self: Sized,
    {
        unsafe {
            UserGroup(
                with_uninitialized(|newgroup| {
                    ffi::MPI_Group_union(self.as_raw(), other.as_raw(), newgroup)
                })
                .1,
            )
        }
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
        Self: Sized,
    {
        unsafe {
            UserGroup(
                with_uninitialized(|newgroup| {
                    ffi::MPI_Group_intersection(self.as_raw(), other.as_raw(), newgroup)
                })
                .1,
            )
        }
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
        Self: Sized,
    {
        unsafe {
            UserGroup(
                with_uninitialized(|newgroup| {
                    ffi::MPI_Group_difference(self.as_raw(), other.as_raw(), newgroup)
                })
                .1,
            )
        }
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
        unsafe {
            UserGroup(
                with_uninitialized(|newgroup| {
                    ffi::MPI_Group_incl(self.as_raw(), ranks.count(), ranks.as_ptr(), newgroup)
                })
                .1,
            )
        }
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
        unsafe {
            UserGroup(
                with_uninitialized(|newgroup| {
                    ffi::MPI_Group_excl(self.as_raw(), ranks.count(), ranks.as_ptr(), newgroup)
                })
                .1,
            )
        }
    }

    /// Number of processes in the group.
    ///
    /// # Standard section(s)
    ///
    /// 6.3.1
    fn size(&self) -> Rank {
        unsafe { with_uninitialized(|size| ffi::MPI_Group_size(self.as_raw(), size)).1 }
    }

    /// Rank of this process within the group.
    ///
    /// # Standard section(s)
    ///
    /// 6.3.1
    fn rank(&self) -> Option<Rank> {
        unsafe {
            let (_, rank) = with_uninitialized(|rank| ffi::MPI_Group_rank(self.as_raw(), rank));
            if rank == ffi::RSMPI_UNDEFINED {
                None
            } else {
                Some(rank)
            }
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
        Self: Sized,
    {
        unsafe {
            let (_, translated) = with_uninitialized(|translated| {
                ffi::MPI_Group_translate_ranks(self.as_raw(), 1, &rank, other.as_raw(), translated)
            });
            if translated == ffi::RSMPI_UNDEFINED {
                None
            } else {
                Some(translated)
            }
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
        Self: Sized,
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
        Self: Sized,
    {
        unsafe {
            with_uninitialized(|relation| {
                ffi::MPI_Group_compare(self.as_raw(), other.as_raw(), relation)
            })
            .1
            .into()
        }
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
        if i == unsafe { ffi::RSMPI_IDENT } {
            return GroupRelation::Identical;
        } else if i == unsafe { ffi::RSMPI_SIMILAR } {
            return GroupRelation::Similar;
        } else if i == unsafe { ffi::RSMPI_UNEQUAL } {
            return GroupRelation::Unequal;
        }
        panic!("Unknown group relation: {}", i)
    }
}

/// Attributes are keys to user data that can be owned by communicators and
/// accessed by users. They are useful when libraries pass communicators to a
/// different library and get it back in a callback.
///
/// # Standard section(s)
///
/// 7.7.1
pub trait Attribute: AsRaw<Raw = c_int> {
    /// The type an attribute carries.
    type Target;
}

/// Predefined attributes
#[derive(Copy, Clone)]
pub struct SystemAttribute(c_int);

impl SystemAttribute {
    /// Wraps the raw value without checking for null handle
    /// Convert raw integer attribute key into SystemAttribute
    pub unsafe fn from_raw_unchecked(val: c_int) -> Self {
        Self(val)
    }
}

unsafe impl AsRaw for SystemAttribute {
    type Raw = c_int;
    fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Attribute for SystemAttribute {
    type Target = c_int;
}
