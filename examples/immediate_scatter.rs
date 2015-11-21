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

    let mut x = 0 as Rank;
    if rank == root_rank {
        let v = (0..size).collect::<Vec<_>>();
        let req = root_process.immediate_scatter_into_root(&v[..], &mut x);
        req.wait();
    } else {
        let req = root_process.immediate_scatter_into(&mut x);
        req.wait();
    }
    assert_eq!(x, rank);
}
