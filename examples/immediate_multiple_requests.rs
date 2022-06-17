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
    mpi::request::multiple_scope(2 * COUNT, |scope, coll| {
        let prev_proc = if rank > 0 { rank - 1 } else { size - 1 };
        let next_proc = (rank + 1) % size;
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
        let prev_proc = if rank > 0 { rank - 1 } else { size - 1 };
        let next_proc = (rank + 1) % size;
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
        let mut some_buf: Vec<Option<(usize, Status, &i32)>> = vec![None; 2 * COUNT];
        while coll.incomplete() > 0 {
            let count = coll.wait_some(&mut some_buf);
            println!("wait_some(): {} request(s) completed", count);
            assert!(some_buf.iter().any(|elm| elm.is_some()));
            for elm in some_buf.iter() {
                if let Some((_, _, result)) = elm {
                    if **result == rank {
                        send_count += 1;
                    } else {
                        recv_count += 1;
                    }
                }
            }
        }
        assert_eq!(send_count, COUNT);
        assert_eq!(recv_count, COUNT);
    });

    let mut result: Vec<i32> = vec![0; COUNT];
    // Test wait_all()
    mpi::request::multiple_scope(2 * COUNT, |scope, coll| {
        let prev_proc = if rank > 0 { rank - 1 } else { size - 1 };
        let next_proc = (rank + 1) % size;
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
        let mut out: Vec<Option<(usize, Status, &i32)>> = vec![None; 2 * COUNT];
        coll.wait_all(&mut out);
        let mut send_count = 0;
        let mut recv_count = 0;
        for elm in out {
            let (_, _, result) = elm.unwrap();
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
    mpi::request::multiple_scope(COUNT, |scope, coll| {
        let prev_proc = if rank > 0 { rank - 1 } else { size - 1 };
        let next_proc = (rank + 1) % size;
        for elm in &x {
            let sreq = world
                .process_at_rank(next_proc)
                .immediate_send(scope, elm);
            coll.add(sreq);
        }
        for val in result.iter_mut() {
            let rreq = world
                .process_at_rank(prev_proc)
                .immediate_receive_into(scope, val);
            coll.add(rreq);
        }
        let mut out: Vec<Option<(usize, Status, &i32)>> = vec![None; 2 * COUNT];
        coll.wait_all(&mut out);
        assert!(out.iter().all(|elm| elm.is_some()));
    });
    // Ensure the result and x are an incrementing array of integers
    result.sort();
    assert_eq!(result, x);
}
