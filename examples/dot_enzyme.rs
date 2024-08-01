#![deny(warnings)]
#![feature(autodiff)]

use mpi::{topology::SimpleCommunicator, traits::*};

#[autodiff(b_dot_local, Reverse, Duplicated, Duplicated, Active)]
fn dot_local(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(x, y)| x * y).sum()
}

#[inline(never)]
#[autodiff(b_dot_parallel, Reverse, Const, Duplicated, Duplicated, Active)]
fn dot_parallel(comm: &SimpleCommunicator, x: &[f64], y: &[f64]) -> f64 {
    let r_loc = dot_local(x, y);
    let mut r = 0.0_f64;
    comm.all_reduce_sum_into(&r_loc, &mut r);
    r
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let n_loc = 5;
    let x = (0..n_loc)
        .map(|i| f64::from(10 * world.rank() + i))
        .collect::<Vec<_>>();
    let y = (0..n_loc)
        .map(|i| f64::from(100 * world.rank() + i))
        .collect::<Vec<_>>();
    let r = dot_local(&x, &y);
    println!("[{}] local: {}", world.rank(), r);

    let root = world.process_at_rank(0);
    let r = dot_parallel(&world, &x, &y);
    if root.is_self() {
        println!("global: {}", r);
    }

    let mut bx = vec![0.0; n_loc as usize];
    let mut by = vec![0.0; n_loc as usize];
    if true {
        let r = b_dot_parallel(&world, &x, &mut bx, &y, &mut by, 1.0);
        if root.is_self() {
            println!("global: {}", r);
        }
        println!("[{}] bx: {:?}, by: {:?}", world.rank(), bx, by);
    } else {
        let _r = b_dot_local(&x, &mut bx, &y, &mut by, 1.0);
        println!("[{}] bx: {:?}, by: {:?}", world.rank(), bx, by);
    }
}
