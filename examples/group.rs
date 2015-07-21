extern crate mpi;

use mpi::traits::*;

use mpi::topology::GroupRelation;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let g1 = world.group();
    let g2 = world.group();

    assert_eq!(GroupRelation::Identical, g1.compare(&g2));
}
