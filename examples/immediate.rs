extern crate mpi;

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let x = 3.1415f32;
    let mut y: f32 = 0.0;

    {
        let mut sreq = world.this_process().immediate_send(&x);
        let rreq = world.immediate_receive_into(&mut y);
        rreq.wait();
        loop {
            match sreq.test() {
                Ok(_) => { break; }
                Err(req) => { sreq = req; }
            }
        }
    }
    assert_eq!(x, y);

    y = 0.0;
    {
        let rreq = world.immediate_receive_into(&mut y);
        let sreq = world.this_process().immediate_ready_send(&x);
        rreq.wait();
        sreq.wait();
    }
    assert_eq!(x, y);

    assert!(world.immediate_probe().is_none());
    assert!(world.immediate_matched_probe().is_none());

    y = 0.0;
    {
        let sreq = world.this_process().immediate_synchronous_send(&x);
        let preq = world.immediate_matched_probe();
        assert!(preq.is_some());
        let (msg, _) = preq.unwrap();
        let rreq = msg.immediate_matched_receive_into(&mut y);
        rreq.wait();
        sreq.wait();
    }
    assert_eq!(x, y);
}
