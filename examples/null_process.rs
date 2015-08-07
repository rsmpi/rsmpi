extern crate mpi;

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let (msg, status) = world.null_process().receive::<u64>();
    assert_eq!(None, msg);
    assert_eq!(mpi::ffi::RSMPI_PROC_NULL, status.source_rank());

    let x = 1u64;
    world.null_process().send(&x);

    let (mut msg, status) = world.null_process().matched_probe();
    assert!(msg.is_no_proc());
    assert_eq!(mpi::ffi::RSMPI_PROC_NULL, status.source_rank());

    let (msg, status) = msg.matched_receive::<f64>();
    assert_eq!(None, msg);
    assert_eq!(mpi::ffi::RSMPI_PROC_NULL, status.source_rank());
}
