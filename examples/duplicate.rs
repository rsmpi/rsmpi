#![deny(warnings)]

use mpi::{topology::CommunicatorRelation, traits::*};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let moon = world.duplicate();

    world.barrier();
    moon.barrier();

    assert_eq!(CommunicatorRelation::Congruent, world.compare(&moon));
}
