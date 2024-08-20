#![deny(warnings)]
#![allow(clippy::float_cmp)]

use mpi::{
    request::{CancelGuard, WaitGuard},
    traits::*,
};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let x = std::f32::consts::PI;
    let mut y: f32 = 0.0;

    mpi::request::scope(|scope| {
        let mut sreq = world.this_process().immediate_send(scope, &x);
        let rreq = world.any_process().immediate_receive_into(scope, &mut y);
        rreq.wait();
        loop {
            match sreq.test() {
                Ok(_) => {
                    break;
                }
                Err(req) => {
                    sreq = req;
                }
            }
        }
    });
    assert_eq!(x, y);

    y = 0.0;
    mpi::request::scope(|scope| {
        let _rreq = WaitGuard::from(world.any_process().immediate_receive_into(scope, &mut y));
        // WARNING: the *ready* send is *only* permissible here because we're
        // sending to self. Use of *ready* would be a race condition and thus
        // erroneous otherwise.
        //
        // One would typically use `immediate_send` to avoid the unsafety.
        let _sreq =
            WaitGuard::from(unsafe { world.this_process().immediate_ready_send(scope, &x) });
    });
    assert_eq!(x, y);

    assert!(world.any_process().immediate_probe().is_none());
    assert!(world.any_process().immediate_matched_probe().is_none());

    y = 0.0;
    mpi::request::scope(|scope| {
        let _sreq: WaitGuard<_, _> = world
            .this_process()
            .immediate_synchronous_send(scope, &x)
            .into();
        let (msg, _) = loop {
            // Spin for message availability. There is no guarantee that
            // immediate sends, even to the same process, will be immediately
            // visible to an immediate probe.
            let preq = world.any_process().immediate_matched_probe();
            if let Some(p) = preq {
                break p;
            }
        };
        let _rreq: WaitGuard<_, _> = msg.immediate_matched_receive_into(scope, &mut y).into();
    });
    assert_eq!(x, y);

    let future = world.any_process().immediate_receive();
    world.this_process().send(&x);
    let (msg, _) = future.get();
    assert_eq!(x, msg);

    let future = world.any_process().immediate_receive();
    let res = future.r#try();
    assert!(res.is_err());
    let mut future = res.err().unwrap();
    world.this_process().send(&x);
    loop {
        match future.r#try() {
            Ok((msg, _)) => {
                assert_eq!(x, msg);
                break;
            }
            Err(f) => {
                future = f;
            }
        }
    }

    mpi::request::scope(|scope| {
        let sreq = world.this_process().immediate_send(scope, &x);
        sreq.cancel();
        sreq.wait();

        let _sreq = CancelGuard::from(world.this_process().immediate_receive_into(scope, &mut y));
    });
}
