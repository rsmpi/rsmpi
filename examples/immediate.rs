#![deny(warnings)]
#![allow(clippy::float_cmp)]
extern crate mpi;

use mpi::request::{CancelGuard, LocalScope, WaitGuard};
use mpi::traits::*;

use std::pin::Pin;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let x = std::f32::consts::PI;
    let mut y: f32 = 0.0;

    {
        // Scopes can be defined using the `define_scope!` macro. This is the preferred method.
        // A scope should be dropped as soon as the requests attached to it complete so that
        // any borrowed state can be used.
        mpi::define_scope!(scope);

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
    }

    assert_eq!(x, y);

    y = 0.0;
    {
        // Scopes can also be pinned to the heap, though this has some overhead. This scope must
        // be passed by reference, or you must convert it into a `Pin<&LocalScope>` using
        // `as_ref()`.
        let scope: Pin<Box<LocalScope>> = LocalScope::pinned();

        let _rreq = WaitGuard::from(world.any_process().immediate_receive_into(&scope, &mut y));

        // Converts to a `Pin<&LocalScope>`, which can be passed directly to immediate routines.
        let scope: Pin<&LocalScope> = scope.as_ref();
        let _sreq = WaitGuard::from(world.this_process().immediate_ready_send(scope, &x));
    }
    assert_eq!(x, y);

    assert!(world.any_process().immediate_probe().is_none());
    assert!(world.any_process().immediate_matched_probe().is_none());

    y = 0.0;
    // last of all, you can use the `scope` routine
    mpi::request::scope(|scope| {
        let _sreq: WaitGuard<_> = world
            .this_process()
            .immediate_synchronous_send(scope, &x)
            .into();
        let preq = world.any_process().immediate_matched_probe();
        assert!(preq.is_some());
        let (msg, _) = preq.unwrap();
        let _rreq: WaitGuard<_> = msg.immediate_matched_receive_into(scope, &mut y).into();
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

    mpi::define_scope!(scope);

    let sreq = world.this_process().immediate_send(scope, &x);
    sreq.cancel();
    sreq.wait();

    let _sreq = CancelGuard::from(world.this_process().immediate_receive_into(scope, &mut y));
}
