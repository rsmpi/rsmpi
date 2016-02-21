// #![feature(plugin, custom_attribute)]
// #![plugin(c_import)]

#![cfg_attr(feature="clippy", feature(plugin))]
#![cfg_attr(feature="clippy", plugin(clippy))]
#![cfg_attr(feature="clippy", warn(cast_possible_truncation))]
#![cfg_attr(feature="clippy", warn(cast_possible_wrap))]
#![cfg_attr(feature="clippy", warn(cast_precision_loss))]
#![cfg_attr(feature="clippy", warn(cast_sign_loss))]
#![cfg_attr(feature="clippy", warn(mut_mut))]
#![cfg_attr(feature="clippy", warn(mutex_integer))]
#![cfg_attr(feature="clippy", warn(non_ascii_literal))]
#![cfg_attr(feature="clippy", warn(option_unwrap_used))]
#![cfg_attr(feature="clippy", warn(result_unwrap_used))]
#![cfg_attr(feature="clippy", warn(string_add))]
#![cfg_attr(feature="clippy", warn(string_add_assign))]
#![cfg_attr(feature="clippy", warn(unicode_not_nfc))]
#![cfg_attr(feature="clippy", warn(wrong_pub_self_convention))]

#![deny(missing_docs)]

//! Message Passing Interface bindings for Rust
//!
//! The [Message Passing Interface][MPI] (MPI) is a specification for a
//! message-passing style concurrency library. Implementations of MPI are often used to structure
//! parallel computation on High Performance Computing systems. The MPI specification describes
//! bindings for the C programming language (and through it C++) as well as for the Fortran
//! programming language. This library tries to bridge the gap into a more rustic world.
//!
//! [MPI]: http://www.mpi-forum.org
//!
//! # Usage
//!
//! Add the `mpi` crate as a dependency in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! mpi = { git = "https://github.com/bsteinb/rsmpi.git", tag = "0.4.0" }
//! ```
//!
//! Then use it in your program like this:
//!
//! ```no_run
//! extern crate mpi;
//!
//! use mpi::traits::*;
//!
//! fn main() {
//!     let universe = mpi::initialize().unwrap();
//!     let world = universe.world();
//!     let size = world.size();
//!     let rank = world.rank();
//!
//!     if size != 2 {
//!         panic!("Size of MPI_COMM_WORLD must be 2, but is {}!", size);
//!      }
//!
//!     match rank {
//!         0 => {
//!             let msg = vec![4.0f64, 8.0, 15.0];
//!             world.process_at_rank(rank + 1).send(&msg[..]);
//!         }
//!         1 => {
//!             let (msg, status) = world.any_process().receive_vec::<f64>();
//!             println!("Process {} got message {:?}.\nStatus is: {:?}", rank, msg, status);
//!         }
//!         _ => unreachable!()
//!     }
//! }
//! ```
//!
//! # Features
//!
//! The bindings follow the MPI 3.1 specification.
//!
//! Currently supported:
//!
//! - **Groups, Contexts, Communicators**:
//!   - Group and (Intra-)Communicator management from section 6 is mostly complete.
//!   - no Inter-Communicators
//!   - no process topologies
//! - **Point to point communication**:
//!   - standard, buffered, synchronous and ready mode send in blocking and non-blocking variants
//!   - receive in blocking and non-blocking variants
//!   - send-receive
//!   - probe
//!   - matched probe/receive
//! - **Collective communication**:
//!   - barrier
//!   - broadcast
//!   - (all) gather
//!   - scatter
//!   - all to all
//!   - varying counts operations
//!   - reductions/scans
//!   - blocking and non-blocking variants
//! - **Datatypes**: Bridging between Rust types and MPI basic types as well as custom MPI datatypes
//! which can act as views into buffers.
//!
//! Not supported (yet):
//!
//! - Process management
//! - One-sided communication (RMA)
//! - MPI parallel I/O
//! - A million small things
//!
//! The sub-modules contain a more detailed description of which features are and are not
//! supported.
//!
//! # Further Reading
//!
//! While every publicly defined item in this crate should have some documentation attached to it,
//! most of the descriptions are quite terse for now and to the uninitiated will only make sense in
//! combination with the [MPI specification][MPIspec].
//!
//! [MPIspec]: http://www.mpi-forum.org/docs/docs.html

use std::mem;
use std::string::FromUtf8Error;
use std::os::raw::{c_char, c_int};

extern crate conv;

use conv::ConvUtil;

/// The raw C language MPI API
///
/// Documented in the [Message Passing Interface specification][spec]
///
/// [spec]: http://www.mpi-forum.org/docs/docs.html
#[allow(missing_docs, dead_code, non_snake_case, non_camel_case_types)]
pub mod ffi;

pub mod collective;
pub mod datatype;
pub mod point_to_point;
pub mod raw;
pub mod request;
pub mod topology;

/// Re-exports all traits.
pub mod traits {
    pub use collective::traits::*;
    pub use datatype::traits::*;
    pub use point_to_point::traits::*;
    pub use raw::traits::*;
    pub use request::traits::*;
    pub use topology::traits::*;
}

#[doc(inline)]
pub use topology::{initialize, initialize_with_threading, Threading};

use ffi::MPI_Aint;

/// Encodes error values returned by MPI functions.
pub type Error = c_int;
/// Encodes number of values in multi-value messages.
pub type Count = c_int;
/// Can be used to tag messages on the sender side and match on the receiver side.
pub type Tag = c_int;
/// An address in memory
pub type Address = MPI_Aint;

/// Identifies the version of the MPI standard implemented by the library.
///
/// Returns a tuple of `(version, subversion)`, e.g. `(3, 0)`.
///
/// Can be called without initializing MPI.
pub fn get_version() -> (c_int, c_int) {
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
pub fn get_library_version() -> Result<String, FromUtf8Error> {
    let bufsize = ffi::RSMPI_MAX_LIBRARY_VERSION_STRING.value_as().expect(
        &format!("MPI_MAX_LIBRARY_SIZE ({}) cannot be expressed as a usize.",
            ffi::RSMPI_MAX_LIBRARY_VERSION_STRING)
        );
    let mut buf = vec![0u8; bufsize];
    let mut len: c_int = 0;

    unsafe {
        ffi::MPI_Get_library_version(buf.as_mut_ptr() as *mut c_char, &mut len);
    }
    buf.truncate(len.value_as().expect(&format!("Length of library version string ({}) cannot \
                                                 be expressed as a usize.",
                                                len)));
    String::from_utf8(buf)
}
