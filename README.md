# MPI bindings for Rust

[![Travis build status][travis-shield]][travis] [![Documentation: hosted][doc-shield]][doc] [![License: MIT][license-shield]][license] [![crate on crates.io][crate-shield]][crate]

The [Message Passing Interface][MPI] (MPI) is a specification for a
message-passing style concurrency library. Implementations of MPI are often used to structure
parallel computation on High Performance Computing systems. The MPI specification describes
bindings for the C programming language (and through it C++) as well as for the Fortran
programming language. This library tries to bridge the gap into a more rustic world.

[travis-shield]: https://img.shields.io/travis/bsteinb/rsmpi.svg?style=flat-square
[travis]: https://travis-ci.org/bsteinb/rsmpi
[doc-shield]: https://img.shields.io/badge/documentation-hosted-blue.svg?style=flat-square
[doc]: http://bsteinb.github.io/rsmpi/
[license-shield]: https://img.shields.io/github/license/bsteinb/rsmpi.svg?style=flat-square
[license]: https://github.com/bsteinb/rsmpi/blob/master/LICENSE
[crate-shield]: https://img.shields.io/crates/v/mpi.svg?style=flat-square
[crate]: https://crates.io/crates/mpi
[MPI]: http://www.mpi-forum.org

## Requirements

An implementation of the C language interface of MPI-3.0. These bindings are currently tested with:

- [OpenMPI][OpenMPI] 1.8.8, 1.10.0
- [MPICH][MPICH] 3.2, 3.1.4

To generate FFI definitions `rsmpi` uses `rust-bindgen` which needs `libclang`. See the [bindgen project page][bindgen] for troubleshooting.

[OpenMPI]: https://www.open-mpi.org
[MPICH]: https://www.mpich.org
[bindgen]: https://github.com/crabtw/rust-bindgen

## Building

```
cargo build
```

## Usage

Add the `mpi` crate as a dependency in your `Cargo.toml`:

```toml
[dependencies]
mpi = "0.2"
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
    let _sreq = WaitGuard::from(world.process_at_rank(next_rank).immediate_send(&msg[..]));

    let (msg, status) = world.receive_vec();
    let msg = msg.unwrap();

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
}
```

## Features

The bindings follow the MPI 3.1 specification.

Currently supported:

- **Groups, Contexts, Communicators**:
  - Group and (Intra-)Communicator management from section 6 is mostly complete.
  - no Inter-Communicators
  - noprocess topologies
- **Point to point communication**:
  - standard, buffered, synchronous and ready mode send in blocking and non-blocking variants
  - receive in blocking and non-blocking variants
  - send-receive
  - probe
  - matched probe/receive
- **Collective communication**:
  - barrier in blocking and non-blocking variants
  - broadcast
  - (all) gather
  - scatter
  - all to all
  - no varying counts operations
  - no reductions/scans
- **Datatypes**: Bridging between Rust types and MPI basic types as well as custom MPI datatypes which can act as views into buffers.

Not supported (yet):

- Process management
- One-sided communication (RMA)
- MPI parallel I/O
- A million small things

## Documentation

```
cargo doc
```

Or see the [hosted documentation][doc].

## Examples

See files in [examples/][examples].

[examples]: https://github.com/bsteinb/rsmpi/tree/master/examples

## License

The MIT license, see the file [LICENSE][license].
