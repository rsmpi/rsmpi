extern crate mpi;

use mpi::traits::*;
use mpi::topology::{Rank};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();
    let root_rank = 0;
    let root_process = world.process_at_rank(root_rank);

    let v = if rank == root_rank { Some((0..size).collect::<Vec<_>>()) } else { None };
    let mut x = 0 as Rank;

    root_process.scatter_into(v.as_ref().map(|x| &x[..]), &mut x);

    assert_eq!(x, rank);
}
