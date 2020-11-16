#![deny(warnings)]
extern crate mpi;

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let rank = world.rank();
    let count = world.size() as usize;

    let mut a = vec![mpi::Bool::default(); count];
    world.all_gather_into(&(rank % 2 == 0), &mut a[..]);

    let answer: Vec<mpi::Bool> = (0..count).map(|i| (i % 2 == 0).into()).collect();

    assert_eq!(answer, a);
}
