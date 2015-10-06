extern crate mpi;

use mpi::traits::*;
use mpi::topology::Rank;
use mpi::collective::{self, SystemOperation};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();
    let root_rank = 0;

    let mut sum: Rank = 0;

    world.process_at_rank(root_rank).reduce_into(&rank, Some(&mut sum), SystemOperation::sum());

    if rank == root_rank {
        assert_eq!(sum, size * (size - 1) / 2);
    }

    let mut max: Rank = -1;

    world.all_reduce_into(&rank, &mut max, SystemOperation::max());
    assert_eq!(max, size - 1);

    let a: u64 = 0b0000111111110000;
    let b: u64 = 0b0011110000111100;

    let mut c = b;
    collective::reduce_local_into(&a, &mut c, SystemOperation::bitwise_and());
    assert_eq!(c, 0b0000110000110000);

    let mut d = b;
    collective::reduce_local_into(&a, &mut d, SystemOperation::bitwise_or());
    assert_eq!(d, 0b0011111111111100);

    let mut e = b;
    collective::reduce_local_into(&a, &mut e, SystemOperation::bitwise_xor());
    assert_eq!(e, 0b0011001111001100);
}
