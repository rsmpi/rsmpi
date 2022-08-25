//! MPI error handling and constants

use std::convert::TryInto;
use std::os::raw::c_int;

use crate::ffi;

/// MPI_SUCCESS constant, cast as a c_int here for easier checking of MPI return values
pub const MPI_SUCCESS: c_int = ffi::MPI_SUCCESS as c_int;

macro_rules! build_error_kind {
    {
        $(#[$doc:meta])*
        pub enum $name:ident {
            $(
                 #[$err_doc:meta]
                 #[err($mpi_err:ident)]
                 $rust_err:ident,
            )*
        }
    } => {
        use crate::ffi::{
            $(
            $mpi_err,
            )*
        };

        $(#[$doc])*
        #[derive(Debug, Clone, Copy)]
        pub enum $name {
            $(
            #[$err_doc]
            $rust_err,
            )*
        }

        impl $name {
            /// Convert a raw MPI returncode into an error class, as a Rust enum.
            pub(crate) fn from_raw(err: c_int) -> Option<$name> {
                let mut err_class: c_int = 0;
                let res = unsafe {
                    ffi::MPI_Error_class(err, &mut err_class)
                };
                // If we fail here, then there's something really wrong
                assert_eq!(res, MPI_SUCCESS);
                $(
                if err == $mpi_err.try_into().unwrap() {
                    return Some($name::$rust_err)
                }
                )*
                None
            }
        }
    }
}

build_error_kind! {
    /// Set of error classes that are possible in MPI.
    ///
    /// Note that the docstrings have been copied from the MPI 4.0 standard
    /// document: https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf (9.4)
    pub enum ErrorKind {
        /// Permission denied
        #[err(MPI_ERR_ACCESS)]
        Access,
        /// Errors related to the amode passed to MPI_FILE_OPEN
        #[err(MPI_ERR_AMODE)]
        Amode,
        /// Invalid argument of some other kind
        #[err(MPI_ERR_ARG)]
        Arg,
        /// Invalid assertion argument
        #[err(MPI_ERR_ASSERT)]
        Assert,
        /// Invalid file name (e.g., path name too long)
        #[err(MPI_ERR_BAD_FILE)]
        BadFile,
        /// Invalid base passed to MPI_FREE_MEM
        #[err(MPI_ERR_BASE)]
        Base,
        /// Invalid buffer pointer argument
        #[err(MPI_ERR_BUFFER)]
        Buffer,
        /// Invalid communicator argument
        #[err(MPI_ERR_COMM)]
        Comm,
        /// An error occurred in a user supplied data conversion function
        #[err(MPI_ERR_CONVERSION)]
        Conversion,
        /// Invalid count argument
        #[err(MPI_ERR_COUNT)]
        Count,
        /// Invalid dimension argument
        #[err(MPI_ERR_DIMS)]
        Dims,
        /// Invalid displacement argument
        #[err(MPI_ERR_DISP)]
        Disp,
        /// Conversion functions could not be registered because a data representation identifier that was already defined was passed to MPI_REGISTER_DATAREP
        #[err(MPI_ERR_DUP_DATAREP)]
        DupDatarep,
        /// Invalid file handle argument
        #[err(MPI_ERR_FILE)]
        File,
        /// File exists
        #[err(MPI_ERR_FILE_EXISTS)]
        FileExists,
        /// File operation could not be completed, as the file is currently open by some process
        #[err(MPI_ERR_FILE_IN_USE)]
        FileInUse,
        /// Invalid group argument
        #[err(MPI_ERR_GROUP)]
        Group,
        /// Invalid info argument
        #[err(MPI_ERR_INFO)]
        Info,
        /// Key longer than MPI_MAX_INFO_KEY
        #[err(MPI_ERR_INFO_KEY)]
        InfoKey,
        /// Invalid key passed to MPI_INFO_DELETE
        #[err(MPI_ERR_INFO_NOKEY)]
        InfoNokey,
        /// Value longer than MPI_MAX_INFO_VAL
        #[err(MPI_ERR_INFO_VALUE)]
        InfoValue,
        /// Error code is in status
        #[err(MPI_ERR_IN_STATUS)]
        Status,
        /// Internal MPI (implementation) error
        #[err(MPI_ERR_INTERN)]
        Intern,
        /// Other I/O error
        #[err(MPI_ERR_IO)]
        IO,
        /// Invalid keyval argument
        #[err(MPI_ERR_KEYVAL)]
        Keyval,
        /// Invalid service name passed to MPI_LOOKUP_NAME
        #[err(MPI_ERR_NAME)]
        Name,
        /// MPI_ALLOC_MEM failed because memory is exhausted
        #[err(MPI_ERR_NO_MEM)]
        NoMem,
        /// Not enough space
        #[err(MPI_ERR_NO_SPACE)]
        NoSpace,
        /// File does not exist
        #[err(MPI_ERR_NO_SUCH_FILE)]
        NoSuchFile,
        /// Collective argument not identical on all processes, or collective routines called in a different order by different processes
        #[err(MPI_ERR_NOT_SAME)]
        NotSame,
        /// Invalid operation argument
        #[err(MPI_ERR_OP)]
        Op,
        /// Known error not in this list
        #[err(MPI_ERR_OTHER)]
        Other,
        /// Pending request
        #[err(MPI_ERR_PENDING)]
        Pending,
        /// Invalid port name passed to MPI_COMM_CONNECT
        #[err(MPI_ERR_PORT)]
        Port,
    /*
        /// Operation failed because a peer process has aborted
        #[err(MPI_ERR_PROC_ABORTED)]
        ProcAborted,
    */
        /// Quota exceeded
        #[err(MPI_ERR_QUOTA)]
        Quota,
        /// Invalid rank argument
        #[err(MPI_ERR_RANK)]
        Rank,
        /// Read-only file or file system
        #[err(MPI_ERR_READ_ONLY)]
        ReadOnly,
        /// Invalid request argument
        #[err(MPI_ERR_REQUEST)]
        Request,
        /// Memory cannot be attached (e.g., because of resource exhaustion)
        #[err(MPI_ERR_RMA_ATTACH)]
        RMAAttach,
        /// Conflicting accesses to window
        #[err(MPI_ERR_RMA_CONFLICT)]
        RMAConflict,
        /// Passed window has the wrong flavor for the called function
        #[err(MPI_ERR_RMA_FLAVOR)]
        RMAFlavor,
        /// Target memory is not part of the window (in the case of a window created with MPI_WIN_CREATE_DYNAMIC, target memory is not attached
        #[err(MPI_ERR_RMA_RANGE)]
        RMARange,
        /// Memory cannot be shared (e.g., some process in the group of the specified communicator cannot expose shared memory)
        #[err(MPI_ERR_RMA_SHARED)]
        RMAShared,
        /// Wrong synchronization of RMA calls
        #[err(MPI_ERR_RMA_SYNC)]
        RMASync,
        /// Invalid root argument
        #[err(MPI_ERR_ROOT)]
        Root,
        /// Invalid service name passed to MPI_UNPUBLISH_NAME
        #[err(MPI_ERR_SERVICE)]
        Service,
    /*
        /// Invalid session argument
        #[err(MPI_ERR_SESSION)]
        Session,
    */
        /// Invalid size argument
        #[err(MPI_ERR_SIZE)]
        Size,
        /// Error in spawning processes
        #[err(MPI_ERR_SPAWN)]
        Spawn,
        /// Invalid tag argument
        #[err(MPI_ERR_TAG)]
        Tag,
        /// Invalid topology argument
        #[err(MPI_ERR_TOPOLOGY)]
        Topology,
        /// Message truncated on receive
        #[err(MPI_ERR_TRUNCATE)]
        Truncate,
        /// Invalid datatype argument
        #[err(MPI_ERR_TYPE)]
        Type,
        /// Unknown error
        #[err(MPI_ERR_UNKNOWN)]
        Unknown,
        /// Unsupported datarep passed to MPI_FILE_SET_VIEW
        #[err(MPI_ERR_UNSUPPORTED_DATAREP)]
        Datarep,
        /// Unsupported operation, such as seeking on a file which supports sequential access only
        #[err(MPI_ERR_UNSUPPORTED_OPERATION)]
        UnsupportedOperation,
    /*
        /// Value is too large to store
        #[err(MPI_ERR_VALUE_TOO_LARGE)]
        ValueTooLarge,
    */
        /// Invalid window argument
        #[err(MPI_ERR_WIN)]
        Win,
        /// Last error code
        #[err(MPI_ERR_LASTCODE)]
        LastCode,
    }
}

/// Crate-internal function for mapping from MPI return code to RSMPI ErrorKind
pub(crate) fn error_kind(res: c_int) -> ErrorKind {
    match ErrorKind::from_raw(res) {
        Some(kind) => kind,
        None => panic!("Could not find matching ErrorKind for returncode '{}'", res),
    }
}
