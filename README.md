# MPI bindings for Rust

[![Travis build status][travis-shield]][travis] [![Documentation: hosted][doc-shield]][doc] [![License: Apache License 2.0 or MIT][license-shield]][license] [![latest GitHub release][release-shield]][release] [![crate on crates.io][crate-shield]][crate]

The [Message Passing Interface][MPI] (MPI) is a specification for a
message-passing style concurrency library. Implementations of MPI are often used to structure
parallel computation on High Performance Computing systems. The MPI specification describes
bindings for the C programming language (and through it C++) as well as for the Fortran
programming language. This library tries to bridge the gap into a more rustic world.

[travis-shield]: https://img.shields.io/travis/bsteinb/rsmpi/master.svg?style=flat-square
[travis]: https://travis-ci.org/bsteinb/rsmpi
[doc-shield]: https://img.shields.io/badge/documentation-hosted-blue.svg?style=flat-square
[doc]: http://bsteinb.github.io/rsmpi/
[license-shield]: https://img.shields.io/badge/license-Apache_License_2.0_or_MIT-blue.svg?style=flat-square
[license]: https://github.com/bsteinb/rsmpi#license
[release-shield]: https://img.shields.io/github/release/bsteinb/rsmpi.svg?style=flat-square
[release]: https://github.com/bsteinb/rsmpi/releases/latest
[crate-shield]: https://img.shields.io/crates/v/mpi.svg?style=flat-square
[crate]: https://crates.io/crates/mpi
[MPI]: http://www.mpi-forum.org

## Requirements

An implementation of the C language interface that conforms to MPI-3.1. `rsmpi` is currently tested with these implementations:

- [OpenMPI][OpenMPI] 2.0.4, 2.1.2, 3.0.0
- [MPICH][MPICH] 3.2.1, 3.1.4
- [MS-MPI (Windows)][MS-MPI] 9.0.1

For a reasonable chance of success with `rsmpi` any MPI implementation that you want to use with it should satisfy the following assumptions that `rsmpi` currently makes:

- The implementation should provide a C compiler wrapper `mpicc`.
- `mpicc -show` should print the full command line that is used to invoke the wrapped C compiler.
- The result of `mpicc -show` contains the libraries, library search paths, and header search paths in a format understood by GCC (e.g. `-lmpi`, `-I/usr/local/include`, ...).

Since the MPI standard leaves some details of the C API unspecified (e.g. whether to implement certain constants and even functions using preprocessor macros or native C constructs, the details of most types, ...) `rsmpi` takes a two step approach to generating functional low-level bindings.

First, it uses a thin static library written in C (see [rsmpi.h][rsmpih] and [rsmpi.c][rsmpic]) that tries to capture the underspecified identifiers and re-exports them with a fixed C API. This library is built from [build.rs][buildrs] using the `gcc` crate.

Second, to generate FFI definitions tailored to each MPI implementation, `rsmpi` uses `rust-bindgen` which needs `libclang`. See the [bindgen project page][bindgen] for more information.

Furthermore, `rsmpi` uses the `libffi` crate which installs the native `libffi` which depends on certain build tools. See the [libffi project page][libffi] for more information.

[OpenMPI]: https://www.open-mpi.org
[MPICH]: https://www.mpich.org
[MS-MPI]: https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi
[rsmpih]: https://github.com/bsteinb/rsmpi/blob/master/src/ffi/rsmpi.h
[rsmpic]: https://github.com/bsteinb/rsmpi/blob/master/src/ffi/rsmpi.c
[buildrs]: https://github.com/bsteinb/rsmpi/blob/master/build.rs
[bindgen]: https://github.com/servo/rust-bindgen
[libffi]: https://github.com/tov/libffi-rs

## Usage

Add the `mpi` crate as a dependency in your `Cargo.toml`:

```toml
[dependencies]
mpi = "0.5"
```

Then use it in your program like this:

```rust
extern crate mpi;

use mpi::traits::*;
use mpi::request::WaitGuard;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();

    let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
    let previous_rank = if rank - 1 >= 0 { rank - 1 } else { size - 1 };

    let msg = vec![rank , 2 * rank, 4 * rank];
    mpi::request::scope(|scope| {
        let _sreq = WaitGuard::from(world.process_at_rank(next_rank).immediate_send(scope, &msg[..]));

        let (msg, status) = world.any_process().receive_vec();

        println!("Process {} got message {:?}.\nStatus is: {:?}", rank, msg, status);
        let x = status.source_rank();
        assert_eq!(x, previous_rank);
        assert_eq!(vec![x, 2 * x, 4 * x], msg);

        let root_rank = 0;
        let root_process = world.process_at_rank(root_rank);

        let mut a;
        if world.rank() == root_rank {
            a = vec![2, 4, 8, 16];
            println!("Root broadcasting value: {:?}.", &a[..]);
        } else {
            a = vec![0; 4];
        }
        root_process.broadcast_into(&mut a[..]);
        println!("Rank {} received value: {:?}.", world.rank(), &a[..]);
        assert_eq!(&a[..], &[2, 4, 8, 16]);
    });
}
```

## Features

The bindings follow the MPI 3.1 specification.

Currently supported:

- **Groups, Contexts, Communicators**:
  - Group and (Intra-)Communicator management from section 6 is mostly complete.
  - no Inter-Communicators
  - no process topologies
- **Point to point communication**:
  - standard, buffered, synchronous and ready mode send in blocking and non-blocking variants
  - receive in blocking and non-blocking variants
  - send-receive
  - probe
  - matched probe/receive
- **Collective communication**:
  - barrier
  - broadcast
  - (all) gather
  - scatter
  - all to all
  - varying counts operations
  - reductions/scans
  - blocking and non-blocking variants
- **Datatypes**: Bridging between Rust types and MPI basic types as well as custom MPI datatypes which can act as views into buffers.

Not supported (yet):

- Process management
- One-sided communication (RMA)
- MPI parallel I/O
- A million small things

## Documentation

Every public item of `rsmpi` should at least have a short piece of documentation associated with it. Documentation can be generated via:

```
cargo doc
```

Documentation for the latest version of the crate released to crates.io is [hosted on Github pages][doc].

## Examples

See files in [examples/][examples]. These examples also act as [integration tests][travis].

[examples]: https://github.com/bsteinb/rsmpi/tree/master/examples

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
