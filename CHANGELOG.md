# RSMPI Release Notes

## `main` branch

**MSRV:** 1.65

### New Features

* [[PR 140]](https://github.com/rsmpi/rsmpi/pull/140) Support for attributes, inter-communicators, and process management.
* [[PR 155]](https://github.com/rsmpi/rsmpi/pull/155) Remove type-distinction between system-communicators and user-communicators.
* [[PR 159]](https://github.com/rsmpi/rsmpi/pull/159) Add `"complex"` feature to support send/receive of complex types from the [`num-complex` crate](https://crates.io/crates/num-complex).

### Maintenance

* More precise optional dependencies for features.
* `build-probe-mpi`: support for Cray compilers and some quoting fixes

## 0.6.0 (2022-08-05)

**MSRV:** 1.54.0

### Notes

In order to use the `user-operations` optional feature (enabled by default), you may want to check
that  you are using at least `libffi-sys` version 1.1.0. This can be done by running `cargo
update`, but for most users this will likely be unnecessary.

### New Features

* [[PR 122]](https://github.com/rsmpi/rsmpi/pull/122) Multiple request completion via `MPI_Waitsome` and friends.
* [[PR 113]](https://github.com/rsmpi/rsmpi/pull/113) Make `Communicator` and `Group` traits object-safe
* [[PR 110]](https://github.com/rsmpi/rsmpi/pull/110) Implement `Buffer` and `BufferMut` for `Vec`,
    allowing cleaner syntax
- [[PR 52]](https://github.com/rsmpi/rsmpi/pull/52) Safely send and receive structs with arbitrary
    data layout, including padding, using `#[derive(Equivalence)]`.
- [[PR 88]](https://github.com/rsmpi/rsmpi/pull/88) Support for `MPI_Waitany` using `mpi::wait_any`.
- [[PR 46]](https://github.com/rsmpi/rsmpi/pull/46)
    [MS-MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi) on Windows
    is now supported.
- [[PR 58]](https://github.com/rsmpi/rsmpi/pull/58) Support for "cartesian" communicators using
    `CartestianCommunicator`
- [[PR 51]](https://github.com/rsmpi/rsmpi/pull/51) Support `MPI_Pack` and `MPI_Unpack` using
    `Communicator::pack`, `Communicator::pack_into`, and `Communicator::unpack_into`.
- [[PR 49]](https://github.com/rsmpi/rsmpi/pull/49) Build compound datatypes without committing
    intermediate types using `UncommittedUserDatatype`.
- [[PR 53]](https://github.com/rsmpi/rsmpi/pull/53) Construct a `UserCommunicator` from an
    `MPI_Comm` with `UserCommunicator::from_raw`.
- [[PR 90]](https://github.com/rsmpi/rsmpi/pull/90) Support `MPI_Comm_split_type` using
    `Communicator::split_shared`, which splits communicators into subcommunicators that are capable
    of creating shared memory regions.
- [[PR 27]](https://github.com/rsmpi/rsmpi/pull/27) Support for `MPI_Comm_get_name` and
    `MPI_Comm_set_name` using `Communicator::set_name` and `Communicator::get_name`.

### Changed APIs
- [[PR 48]](https://github.com/rsmpi/rsmpi/pull/48) `UserDatatype::structured` no longer takes a
    `count` argument.

### Fixed Bugs
- [[PR 77]](https://github.com/rsmpi/rsmpi/pull/77) Failure to complete scoped requests now results
    in immediate process termination, rather than a panic.
- [[PR 53]](https://github.com/rsmpi/rsmpi/pull/96) Portability improved with newer libffi and
    bindgen

### Deprecated APIs

### Removed Features

## Old Releases
Previous releases do not have discretized release notes. Please review the git commmit log to
understand which features were added in each version.
