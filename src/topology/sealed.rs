use crate::ffi;
use crate::topology::comm_is_inter;
use crate::traits::AsRaw;
use mpi_sys::MPI_Comm;

/// A raw communicator handle.
pub enum CommunicatorHandle {
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
    /// * `InterCommunicator` if the handle is an inter-communicator
    /// * `ParentCommunicator` if the handle is the parent communicator
    /// * `UserCommunicator` otherwise.
    ///
    /// # Safety
    /// - `raw` must be a live communicator handle or `MPI_COMM_NULL`
    /// - `raw` must not be used after calling this function
    pub unsafe fn try_from_raw(raw: MPI_Comm) -> Option<CommunicatorHandle> {
        if raw == ffi::RSMPI_COMM_NULL {
            None
        } else if raw == ffi::RSMPI_COMM_WORLD {
            Some(CommunicatorHandle::World)
        } else if raw == ffi::RSMPI_COMM_SELF {
            Some(CommunicatorHandle::SelfComm)
        } else {
            if comm_is_inter(raw) {
                let mut parent_comm = ffi::RSMPI_COMM_NULL;
                ffi::MPI_Comm_get_parent(&mut parent_comm);
                if raw == parent_comm {
                    Some(CommunicatorHandle::Parent(raw))
                } else {
                    Some(CommunicatorHandle::InterComm(raw))
                }
            } else {
                Some(CommunicatorHandle::User(raw))
            }
        }
    }

    /// Create a `CommunicatorHandle::UserCommunicator` rom a raw handle without checking if it is a
    /// null-handle, world-handle, self-handle, or an inter-communicator handle.
    ///
    /// # Safety
    /// - `raw` must be a live communicator handle
    /// - `raw` must not be an inter-communicator handle
    /// - `raw` must not be a system handle (i.e. `MPI_COMM_WORLD` or `MPI_COMM_SELF`)
    /// - `raw` must not be the parent communicator
    /// - `raw` must not be used after calling this function
    pub unsafe fn simple_comm_from_raw(raw: MPI_Comm) -> CommunicatorHandle {
        debug_assert_ne!(raw, ffi::RSMPI_COMM_NULL);
        debug_assert_ne!(raw, ffi::RSMPI_COMM_WORLD);
        debug_assert_ne!(raw, ffi::RSMPI_COMM_SELF);
        debug_assert!(!comm_is_inter(raw));
        CommunicatorHandle::User(raw)
    }

    /// Create a `CommunicatorHandle::InterCommunicator` from a raw handle without checking whether
    /// it is `MPI_COMM_NULL` or if it is actually an inter-comm.
    ///
    /// # Safety
    /// - `raw` must be a live communicator handle
    /// - `raw` must be an inter-communicator handle
    /// - `raw` must not be the parent communicator
    /// - `raw` must not be used after calling this function
    pub unsafe fn inter_comm_from_raw(raw: MPI_Comm) -> CommunicatorHandle {
        debug_assert_ne!(raw, ffi::RSMPI_COMM_NULL);
        debug_assert!(comm_is_inter(raw));
        CommunicatorHandle::InterComm(raw)
    }

    /// Create a `CommunicatorHandle::ParentCommunicator` from a raw handle without checking whether
    /// it is `MPI_COMM_NULL` or if it is actually the parent comm.
    ///
    /// # Safety
    /// - `raw` must be a live communicator handle
    /// - `raw` must be the parent communicator
    /// - `raw` must not be used after calling this function
    #[allow(dead_code)]
    pub unsafe fn parent_comm_from_raw(raw: MPI_Comm) -> CommunicatorHandle {
        debug_assert_ne!(raw, ffi::RSMPI_COMM_NULL);
        debug_assert!({
            let mut parent = ffi::RSMPI_COMM_NULL;
            ffi::MPI_Comm_get_parent(&mut parent);
            raw == parent
        });
        CommunicatorHandle::Parent(raw)
    }

    /// Returns true if the handle is of an inter-comm
    #[allow(dead_code)]
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
            CommunicatorHandle::Parent(_) => { /* not useful to drop (would nullify other references) */
            }
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

/// Get a handle from a communicator. This trait is used internally to treat different
/// communicator types uniformly and allow borrowing handles from trait-object communicators.
pub trait AsHandle: AsRaw<Raw = MPI_Comm> {
    fn as_handle(&self) -> &CommunicatorHandle;
}
