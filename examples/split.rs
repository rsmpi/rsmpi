extern crate mpi;

use mpi::traits::*;
use mpi::topology::{Color};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let underworld = world.split_by_color(Color::with_value(world.rank() % 3)).unwrap();
    underworld.barrier();

    println!("Rank {} of {} on world is rank {} of {} on underworld.",
        world.rank(), world.size(), underworld.rank(), underworld.size());
}
