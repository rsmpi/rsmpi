#![deny(warnings)]
extern crate mpi;

use mpi::traits::*;
use mpi::topology::Rank;
use mpi::collective::{SystemOperation, UserOperation};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();
    let root_rank = 0;

    if rank == root_rank {
        let mut sum: Rank = 0;
        mpi::request::scope(|scope| {
            world
                .process_at_rank(root_rank)
                .immediate_reduce_into_root(scope, &rank, &mut sum, SystemOperation::sum())
                .wait();
        });
        assert_eq!(sum, size * (size - 1) / 2);
    } else {
        mpi::request::scope(|scope| {
            world
                .process_at_rank(root_rank)
                .immediate_reduce_into(scope, &rank, SystemOperation::sum())
                .wait();
        });
    }

    let mut max: Rank = -1;

    mpi::request::scope(|scope| {
        world
            .immediate_all_reduce_into(scope, &rank, &mut max, SystemOperation::max())
            .wait();
    });
    assert_eq!(max, size - 1);

    let a = (0..size).collect::<Vec<_>>();
    let mut b: Rank = 0;

    mpi::request::scope(|scope| {
        world
            .immediate_reduce_scatter_block_into(scope, &a[..], &mut b, SystemOperation::product())
            .wait();
    });
    assert_eq!(b, rank.pow(size as u32));

    let op = UserOperation::commutative(|x, y| {
        let x: &[Rank] = x.downcast().unwrap();
        let y: &mut [Rank] = y.downcast().unwrap();
        for (&x_i, y_i) in x.iter().zip(y) {
            *y_i += x_i;
        }
    });
    let mut c = 0;
    mpi::request::scope(|scope| {
        world
            .immediate_all_reduce_into(scope, &rank, &mut c, &op)
            .wait();
    });
    assert_eq!(c, size * (size - 1) / 2);
}
