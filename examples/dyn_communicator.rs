#![deny(warnings)]
extern crate mpi;

use mpi::collective::SystemOperation;
use mpi::topology::Communicator;
use mpi::traits::{CommunicatorCollectives, Group};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    prefix_sum(&world, "World: ");

    let g1 = world.group().include(&(0..world.size()).filter(|x| x % 2 != 0).collect::<Vec<_>>());
    let g2 = world.group().include(&(0..world.size()).filter(|x| x % 2 == 0).collect::<Vec<_>>());
    world.split_by_subgroup_collective(&g1).map(|c| prefix_sum(&c, "Group 1: "));
    world.split_by_subgroup_collective(&g2).map(|c| prefix_sum(&c, "Group 2: "));
}

// this example mainly tests if the following function compiles
fn prefix_sum(comm: &dyn Communicator, output_prefix: &str) {
    let rank = comm.rank() as usize;
    let mut target = 0;

    comm.scan_into(&rank, &mut target, SystemOperation::sum());

    println!("{} rank {}: {:?}", output_prefix, rank, target);
}