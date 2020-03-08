#![deny(warnings)]
#![allow(clippy::float_cmp)]
extern crate mpi;

use mpi::traits::*;
use std::{thread, time};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let x = std::f32::consts::PI;
    let mut y: f32 = 0.0;

    mpi::request::scope(|scope| {
        if world.rank() == 0 {
            let mut requests = Vec::new();
            for i in 1..world.size() {
                requests.push(
                    world
                        .process_at_rank(i)
                        .immediate_synchronous_send(scope, &x),
                );
            }

            println!("World size {}", world.size());
            for _i in 1..world.size() + 1 {
                let (index, _status) = mpi::request::wait_any(&mut requests);
                if index != mpi_sys::MPI_UNDEFINED {
                    println!("Request with index {} completed", index);
                } else {
                    println!("No requests can be completed");
                }
            }
            println!("All requests completed");
        } else {
            let secs = time::Duration::from_secs((1 * world.rank()) as u64);

            thread::sleep(secs);

            let rreq = world.any_process().immediate_receive_into(scope, &mut y);
            rreq.wait();
            println!("Process {} received data", world.rank());
        }
    });
}
