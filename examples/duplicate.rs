extern crate mpi;

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let moon = world.duplicate();

    world.barrier();
    moon.barrier();
}
