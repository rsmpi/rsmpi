extern crate mpi;

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    println!("Before barrier, rank {}.", world.rank());
    world.barrier();
    println!("After barrier, rank {}.", world.rank());
}
