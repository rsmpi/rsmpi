#![deny(warnings)]
extern crate mpi;

use mpi::traits::*;
use mpi::topology::CommunicatorRelation;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let moon = world.duplicate();

    world.barrier();
    moon.barrier();

    assert_eq!(CommunicatorRelation::Congruent, world.compare(&moon));
}
