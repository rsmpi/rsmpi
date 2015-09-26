// #![feature(plugin, custom_attribute)]
// #![plugin(c_import)]

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
//! mpi = "0.1.8"
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
//!             let (msg, status) = world.receive_vec::<f64>();
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
//!   - noprocess topologies
//! - **Point to point communication**:
//!   - standard, buffered, synchronous and ready mode send in blocking and non-blocking variants
//!   - receive in blocking and non-blocking variants
//!   - send-receive
//!   - probe
//!   - matched probe/receive
//! - **Collective communication**:
//!   - barrier in blocking and non-blocking variants
//!   - broadcast
//!   - (all) gather
//!   - scatter
//!   - all to all
//!   - no varying counts operations
//!   - no reductions/scans
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

extern crate libc;

/// The raw C language MPI API
///
/// Documented in the [Message Passing Interface specification][spec]
///
/// [spec]: http://www.mpi-forum.org/docs/docs.html
#[allow(dead_code, non_snake_case, non_camel_case_types)]
pub mod ffi;

pub mod collective;
pub mod datatype;
pub mod point_to_point;
pub mod topology;
pub mod traits;

pub use topology::{initialize, initialize_with_threading, Threading};

/// Encodes error values returned by MPI functions.
pub type Error = ::libc::c_int;
/// Encodes number of values in multi-value messages.
pub type Count = ::libc::c_int;
/// Can be used to tag messages on the sender side and match on the receiver side.
pub type Tag = ::libc::c_int;
/// An address in memory
pub type Address = ::ffi::MPI_Aint;
