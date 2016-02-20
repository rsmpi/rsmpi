extern crate mpi;

use mpi::traits::*;
use mpi::request::{CancelGuard, WaitGuard};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let x = 3.1415f32;
    let mut y: f32 = 0.0;

    {
        let mut sreq = world.this_process().immediate_send(&x);
        let rreq = world.any_process().immediate_receive_into(&mut y);
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
        let _rreq = WaitGuard::from(world.any_process().immediate_receive_into(&mut y));
        let _sreq = WaitGuard::from(world.this_process().immediate_ready_send(&x));
    }
    assert_eq!(x, y);

    assert!(world.any_process().immediate_probe().is_none());
    assert!(world.any_process().immediate_matched_probe().is_none());

    y = 0.0;
    {
        let _sreq: WaitGuard<_> = world.this_process().immediate_synchronous_send(&x).into();
        let preq = world.any_process().immediate_matched_probe();
        assert!(preq.is_some());
        let (msg, _) = preq.unwrap();
        let _rreq: WaitGuard<_> = msg.immediate_matched_receive_into(&mut y).into();
    }
    assert_eq!(x, y);

    let future = world.any_process().immediate_receive();
    world.this_process().send(&x);
    let (msg, _) = future.get();
    assert_eq!(x, msg);

    let future = world.any_process().immediate_receive();
    let res = future.try();
    assert!(res.is_err());
    let mut future = res.err().unwrap();
    world.this_process().send(&x);
    loop {
        match future.try() {
            Ok((msg, _)) => {
                assert_eq!(x, msg);
                break;
            }
            Err(f) => { future = f; }
        }
    }

    let sreq = world.this_process().immediate_send(&x);
    sreq.cancel();

    let _sreq = CancelGuard::from(world.this_process().immediate_receive_into(&mut y));
}
