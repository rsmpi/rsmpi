# MPI bindings for Rust

## Requirements

An implementation of the C language interface that conforms to MPI-3.1. `rsmpi` is currently tested with these implementations:
- [OpenMPI][OpenMPI] 3.1.4, 4.0.1
- [MPICH][MPICH] 3.1.4

Since the MPI standard leaves some details of the C API unspecified (e.g. whether to implement certain constants and even functions using preprocessor macros or native C constructs, the details of most types, ...) `rsmpi` takes a two step approach to generating functional low-level bindings.

First, it uses a thin static library written in C (see [rsmpi.h][rsmpih] and [rsmpi.c][rsmpic]) that tries to capture the underspecified identifiers and re-exports them with a fixed C API. This library is built from [build.rs][buildrs] using the `gcc` crate.

Second, to generate FFI definitions tailored to each MPI implementation, `rsmpi` uses `rust-bindgen` which needs `libclang`. See the [bindgen project page][bindgen] for more information.

Furthermore, `rsmpi` uses the `libffi` crate which installs the native `libffi` which depends on certain build tools. See the [libffi project page][libffi] for more information.

[OpenMPI]: https://www.open-mpi.org
[MPICH]: https://www.mpich.org
[rsmpih]: https://github.com/rsmpi/rsmpi/blob/master/mpi-sys/src/rsmpi.h
[rsmpic]: https://github.com/rsmpi/rsmpi/blob/master/mpi-sys/src/rsmpi.c
[buildrs]: https://github.com/rsmpi/rsmpi/blob/master/mpi-sys/build.rs
[bindgen]: https://github.com/servo/rust-bindgen
[libffi]: https://github.com/tov/libffi-rs

## Build

If building on a Cray system, use the `cc` wrapper rather than `mpicc`, and need to set.

```bash
export CRAY=1 # set to 0 if not on a cray system.
```

## Usage

Add the `mpi` crate as a dependency in your `Cargo.toml`:

```toml
# "features" is optional
[dependencies]
mpi = { git = "https://github.com/skailasa/rsmpi, branch = "main"}
```

Then use it in your program like this:

```rust
extern crate mpi;

use mpi::request::WaitGuard;
use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();

    let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
    let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };

    let msg = vec![rank, 2 * rank, 4 * rank];
    mpi::request::scope(|scope| {
        let _sreq = WaitGuard::from(
            world
                .process_at_rank(next_rank)
                .immediate_send(scope, &msg[..]),
        );

        let (msg, status) = world.any_process().receive_vec();

        println!(
            "Process {} got message {:?}.\nStatus is: {:?}",
            rank, msg, status
        );
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


### Optional Cargo Features

These optional features can be enabled in your cargo manifest. See the [Usage](#usage) section
above.

`user-operations` enables capturing lambdas and safe creation in `UserOperation`. This feature
requires the `libffi` system library, which is not available on all systems out-of-the-box.

```rust
let mut h = 0;
comm.all_reduce_into(
    &(rank + 1),
    &mut h,
    &UserOperation::commutative(|x, y| {
        let x: &[Rank] = x.downcast().unwrap();
        let y: &mut [Rank] = y.downcast().unwrap();
        for (&x_i, y_i) in x.iter().zip(y) {
            *y_i += x_i;
        }
    }),
);
```

`derive` enables the `Equivalence` derive macro, which makes it easy to send structs
over-the-wire without worrying about safety around padding, and allowing arbitrary datatype
matching between structs with the same field order but different layout.

```rust
#[derive(Equivalence)]
struct MyProgramOpts {
    name: [u8; 100],
    num_cycles: u32,
    material_properties: [f64; 20],
}
```

## Documentation

Every public item of `rsmpi` should at least have a short piece of documentation associated with it. Documentation can be generated via:

```
cargo doc
```

Documentation for the latest version of the crate released to crates.io is [hosted on Github pages][doc].

## Examples

See files in [examples/][examples]. These examples also act as [integration tests][travis].

[examples]: https://github.com/rsmpi/rsmpi/tree/master/examples

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
