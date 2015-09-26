extern crate mpi;

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    if rank > 0 {
        let msg = 1u64;
        world.process_at_rank(0).send(&msg);
        let breq = world.immediate_barrier();
        let msg = 2u64;
        world.process_at_rank(0).send(&msg);
        breq.wait();
        let msg = 3u64;
        world.process_at_rank(0).send(&msg);
    } else {
        let n = (size - 1) as usize;
        let mut buf = vec![0u64; 3 * n];
        for x in buf[0..n].iter_mut() {
            world.receive_into(x);
        }
        let breq = world.immediate_barrier();
        for x in buf[n..2 * n].iter_mut() {
            world.receive_into(x);
        }
        breq.wait();
        for x in buf[2 * n..3 * n].iter_mut() {
            world.receive_into(x);
        }
        println!("{:?}", buf);
        assert!(buf[0..2 * n].iter().all(|&x| { x == 1 || x == 2 }));
        assert!(buf[2 * n..3 * n].iter().all(|&x| { x == 3 }));
    }
}
