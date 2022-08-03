//! Example using request handlers.
use mpi;
use mpi::point_to_point::Status;
use mpi::traits::*;

const COUNT: usize = 256;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();

    let x: i32 = 100;
    let mut y: i32 = 0;

    mpi::request::scope(|scope| {
        let sreq = world.this_process().immediate_send(scope, &x);
        let rreq = world.any_process().immediate_receive_into(scope, &mut y);
        let result = rreq.wait_for_data();
        assert_eq!(*result, x);
        sreq.wait();
    });

    // Test wait_any()
    let mut result: Vec<i32> = vec![0; COUNT];
    let prev_proc = (rank - 1 + size) % size;
    let next_proc = (rank + 1) % size;
    mpi::request::multiple_scope(2 * COUNT, |scope, coll| {
        for _ in 0..result.len() {
            let sreq = world
                .process_at_rank(next_proc)
                .immediate_send(scope, &rank);
            coll.add(sreq);
        }
        for val in result.iter_mut() {
            let rreq = world
                .process_at_rank(prev_proc)
                .immediate_receive_into(scope, val);
            coll.add(rreq);
        }
        let mut send_count = 0;
        let mut recv_count = 0;
        while coll.incomplete() > 0 {
            let (_, _, result) = coll.wait_any().unwrap();
            if *result == rank {
                send_count += 1;
            } else {
                recv_count += 1;
            }
        }
        assert_eq!(send_count, COUNT);
        assert_eq!(recv_count, COUNT);
    });

    let mut result: Vec<i32> = vec![0; COUNT];
    // Test wait_some()
    mpi::request::multiple_scope(2 * COUNT, |scope, coll| {
        for _ in 0..result.len() {
            let sreq = world
                .process_at_rank(next_proc)
                .immediate_send(scope, &rank);
            coll.add(sreq);
        }
        for val in result.iter_mut() {
            let rreq = world
                .process_at_rank(prev_proc)
                .immediate_receive_into(scope, val);
            coll.add(rreq);
        }
        let mut send_count = 0;
        let mut recv_count = 0;
        let mut some_buf = vec![];
        let mut finished = 0;
        while coll.incomplete() > 0 {
            coll.wait_some(&mut some_buf);
            println!("wait_some(): {} request(s) completed", some_buf.len());
            finished += some_buf.len();
            assert_eq!(coll.incomplete(), 2 * COUNT - finished);
            for &(_, _, result) in some_buf.iter() {
                if *result == rank {
                    send_count += 1;
                } else {
                    recv_count += 1;
                }
            }
        }
        assert_eq!(send_count, COUNT);
        assert_eq!(recv_count, COUNT);
    });

    let mut result: Vec<i32> = vec![0; COUNT];
    // Test wait_all()
    mpi::request::multiple_scope(2 * COUNT, |scope, coll| {
        for _ in 0..result.len() {
            let sreq = world
                .process_at_rank(next_proc)
                .immediate_send(scope, &rank);
            coll.add(sreq);
        }
        for val in result.iter_mut() {
            let rreq = world
                .process_at_rank(prev_proc)
                .immediate_receive_into(scope, val);
            coll.add(rreq);
        }

        let mut out = vec![];
        coll.wait_all(&mut out);
        assert_eq!(out.len(), 2 * COUNT);
        let mut send_count = 0;
        let mut recv_count = 0;
        for (_, _, result) in out {
            if *result == rank {
                send_count += 1;
            } else {
                recv_count += 1;
            }
        }
        assert_eq!(send_count, COUNT);
        assert_eq!(recv_count, COUNT);
    });

    // Check wait_*() with a buffer of increasing values
    let x: Vec<i32> = (0..COUNT as i32).collect();
    let mut result: Vec<i32> = vec![0; COUNT];
    mpi::request::multiple_scope(2 * COUNT, |scope, coll| {
        for elm in &x {
            let sreq = world.process_at_rank(next_proc).immediate_send(scope, elm);
            coll.add(sreq);
        }
        for val in result.iter_mut() {
            let rreq = world
                .process_at_rank(prev_proc)
                .immediate_receive_into(scope, val);
            coll.add(rreq);
        }
        let mut out: Vec<(usize, Status, &i32)> = vec![];
        coll.wait_all(&mut out);
        assert_eq!(out.len(), 2 * COUNT);
    });
    // Ensure the result and x are an incrementing array of integers
    result.sort();
    assert_eq!(result, x);
}
