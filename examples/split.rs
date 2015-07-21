extern crate mpi;

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let underworld = world.split(world.rank() % 3);
    underworld.barrier();

    println!("Rank {} of {} on world is rank {} of {} on underworld.",
        world.rank(), world.size(), underworld.rank(), underworld.size());
}
