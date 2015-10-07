extern crate mpi;

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let t_start = universe.get_time();
    world.barrier();
    let t_end = universe.get_time();

    println!("barrier took: {} s", t_end - t_start);
    println!("the clock has a resoltion of {} seconds", universe.get_time_res());
}
