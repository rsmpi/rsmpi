#![deny(warnings)]
extern crate mpi;

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world().unwrap();
    println!(
        "Hello parallel world from process {} of {}!",
        world.rank(),
        world.size()
    );
}
