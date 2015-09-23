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
- [MPICH][MPICH] 3.1.4, 3.0.4

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

```
[dependencies]
mpi = "0.1.8"
```

Then use it in your program like this:

```
extern crate mpi;

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();

    if size != 2 {
        panic!("Size of MPI_COMM_WORLD must be 2, but is {}!", size);
     }

    match rank {
        0 => {
            let msg = vec![4.0f64, 8.0, 15.0];
            world.process_at_rank(rank + 1).send(&msg[..]);
        }
        1 => {
            let (msg, status) = world.receive_vec::<f64>();
            println!("Process {} got message {:?}.\nStatus is: {:?}", rank, msg, status);
        }
        _ => unreachable!()
    }
}
```

## Features

The bindings follow the MPI 3.1 specification.

Currently supported:

- **Groups, Contexts, Communicators**: Only rudimentary features are supported so far.
- **Point to point communication**: Most of the blocking, standard mode functions are supported.
Blocking communication in buffered, synchronous and ready mode are not yet supported.
- **Collective communication**: Blocking barrier, broadcast and gather operations.
- **Datatypes**: Bridging between Rust types and MPI basic types as well as custom MPI datatypes
which can act as views into buffers.

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
