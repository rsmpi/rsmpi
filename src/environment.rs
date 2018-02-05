//! Environmental management
//!
//! This module provides ways for an MPI program to interact with its environment.
//!
//! # Unfinished features
//!
//! - **8.1.2**: `MPI_TAG_UB`, ...
//! - **8.2**: Memory allocation
//! - **8.3, 8.4, and 8.5**: Error handling
use std::{mem, ptr};
use std::cmp::Ordering;
use std::string::FromUtf8Error;
use std::os::raw::{c_char, c_double, c_int, c_void};

use conv::ConvUtil;

use ffi;
use topology::SystemCommunicator;

/// Global context
pub struct Universe {
    buffer: Option<Vec<u8>>,
}

impl Universe {
    /// The 'world communicator'
    ///
    /// Contains all processes initially partaking in the computation.
    ///
    /// # Examples
    /// See `examples/simple.rs`
    pub fn world(&self) -> SystemCommunicator {
        SystemCommunicator::world()
    }

    /// The size in bytes of the buffer used for buffered communication.
    pub fn buffer_size(&self) -> usize {
        self.buffer.as_ref().map_or(0, |buffer| buffer.len())
    }

    /// Set the size in bytes of the buffer used for buffered communication.
    pub fn set_buffer_size(&mut self, size: usize) {
        self.detach_buffer();

        if size > 0 {
            let mut buffer = vec![0; size];
            unsafe {
                ffi::MPI_Buffer_attach(
                    mem::transmute(buffer.as_mut_ptr()),
                    buffer
                        .len()
                        .value_as()
                        .expect("Buffer length exceeds the range of a C int."),
                );
            }
            self.buffer = Some(buffer);
        }
    }

    /// Detach the buffer used for buffered communication.
    pub fn detach_buffer(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            let mut addr: *const c_void = ptr::null();
            let addr_ptr: *mut *const c_void = &mut addr;
            let mut size: c_int = 0;
            unsafe {
                ffi::MPI_Buffer_detach(addr_ptr as *mut c_void, &mut size);
                assert_eq!(addr, mem::transmute(buffer.as_ptr()));
            }
            assert_eq!(
                size,
                buffer
                    .len()
                    .value_as()
                    .expect("Buffer length exceeds the range of a C int.")
            );
        }
    }
}

impl Drop for Universe {
    fn drop(&mut self) {
        self.detach_buffer();
        unsafe {
            ffi::MPI_Finalize();
        }
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
            Single => unsafe_extern_static!(ffi::RSMPI_THREAD_SINGLE),
            Funneled => unsafe_extern_static!(ffi::RSMPI_THREAD_FUNNELED),
            Serialized => unsafe_extern_static!(ffi::RSMPI_THREAD_SERIALIZED),
            Multiple => unsafe_extern_static!(ffi::RSMPI_THREAD_MULTIPLE),
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
        if i == unsafe_extern_static!(ffi::RSMPI_THREAD_SINGLE) {
            return Single;
        } else if i == unsafe_extern_static!(ffi::RSMPI_THREAD_FUNNELED) {
            return Funneled;
        } else if i == unsafe_extern_static!(ffi::RSMPI_THREAD_SERIALIZED) {
            return Serialized;
        } else if i == unsafe_extern_static!(ffi::RSMPI_THREAD_MULTIPLE) {
            return Multiple;
        }
        panic!("Unknown threading level: {}", i)
    }
}

/// Whether the MPI library has been initialized
fn is_initialized() -> bool {
    let mut res: c_int = unsafe { mem::uninitialized() };
    unsafe {
        ffi::MPI_Initialized(&mut res);
    }
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
    initialize_with_threading(Threading::Single).map(|x| x.0)
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
            ffi::MPI_Init_thread(
                ptr::null_mut(),
                ptr::null_mut(),
                threading.as_raw(),
                &mut provided,
            );
        }
        Some((Universe { buffer: None }, provided.into()))
    }
}

/// Level of multithreading supported by this MPI universe
///
/// See the `Threading` enum.
///
/// # Examples
/// See `examples/init_with_threading.rs`
pub fn threading_support() -> Threading {
    let mut res: c_int = unsafe { mem::uninitialized() };
    unsafe {
        ffi::MPI_Query_thread(&mut res);
    }
    res.into()
}

/// Identifies the version of the MPI standard implemented by the library.
///
/// Returns a tuple of `(version, subversion)`, e.g. `(3, 0)`.
///
/// Can be called without initializing MPI.
pub fn version() -> (c_int, c_int) {
    let mut version: c_int = unsafe { mem::uninitialized() };
    let mut subversion: c_int = unsafe { mem::uninitialized() };
    unsafe {
        ffi::MPI_Get_version(&mut version, &mut subversion);
    }
    (version, subversion)
}

/// Describes the version of the MPI library itself.
///
/// Can return an `Err` if the description of the MPI library is not a UTF-8 string.
///
/// Can be called without initializing MPI.
pub fn library_version() -> Result<String, FromUtf8Error> {
    let bufsize = unsafe_extern_static!(ffi::RSMPI_MAX_LIBRARY_VERSION_STRING)
        .value_as()
        .expect(&format!(
            "MPI_MAX_LIBRARY_SIZE ({}) cannot be expressed as a usize.",
            unsafe_extern_static!(ffi::RSMPI_MAX_LIBRARY_VERSION_STRING)
        ));
    let mut buf = vec![0u8; bufsize];
    let mut len: c_int = 0;

    unsafe {
        ffi::MPI_Get_library_version(buf.as_mut_ptr() as *mut c_char, &mut len);
    }
    buf.truncate(len.value_as().expect(&format!(
        "Length of library version string ({}) cannot \
         be expressed as a usize.",
        len
    )));
    String::from_utf8(buf)
}

/// Names the processor that the calling process is running on.
///
/// Can return an `Err` if the processor name is not a UTF-8 string.
pub fn processor_name() -> Result<String, FromUtf8Error> {
    let bufsize = unsafe_extern_static!(ffi::RSMPI_MAX_PROCESSOR_NAME)
        .value_as()
        .expect(&format!(
            "MPI_MAX_LIBRARY_SIZE ({}) \
             cannot be expressed as a \
             usize.",
            unsafe_extern_static!(ffi::RSMPI_MAX_PROCESSOR_NAME)
        ));
    let mut buf = vec![0u8; bufsize];
    let mut len: c_int = 0;

    unsafe {
        ffi::MPI_Get_processor_name(buf.as_mut_ptr() as *mut c_char, &mut len);
    }
    buf.truncate(len.value_as().expect(&format!(
        "Length of processor name string ({}) cannot be \
         expressed as a usize.",
        len
    )));
    String::from_utf8(buf)
}

/// Time in seconds since an arbitrary time in the past.
///
/// The cheapest high-resolution timer available will be used.
pub fn time() -> c_double {
    unsafe { ffi::RSMPI_Wtime() }
}

/// Resolution of timer used in `time()` in seconds
pub fn time_resolution() -> c_double {
    unsafe { ffi::RSMPI_Wtick() }
}
