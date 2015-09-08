extern crate mpi;

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let x = 3.1415f32;
    let mut y: f32 = 0.0;

    {
        let sreq = world.this_process().immediate_send(&x);
        let rreq = world.immediate_receive_into(&mut y);
        rreq.wait();
        sreq.wait();
    }

    assert_eq!(x, y);
}
