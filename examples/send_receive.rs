extern crate mpi;

use mpi::traits::*;
use mpi::topology::Rank;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();

    let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
    let previous_rank = if rank - 1 >= 0 { rank - 1 } else { size - 1 };

    let (msg, status) = world.send_receive::<_, mpi::topology::Rank>(&rank, previous_rank, next_rank);
    assert!(msg.is_some());
    let msg = msg.unwrap();
    println!("Process {} got message {}.\nStatus is: {:?}", rank, msg, status);
    world.barrier();
    assert_eq!(msg, next_rank);

    if rank > 0 {
        let msg = vec![rank, rank + 1, rank - 1];
        world.process_at_rank(0).send(&msg[..]);
    } else {
        for _ in (1..size) {
            let (msg, status) = world.receive_vec::<Rank>();
            assert!(msg.is_some());
            let msg = msg.unwrap();
            println!("Process {} got long message {:?}.\nStatus is: {:?}", rank, msg, status);

            let x = status.source_rank();
            let v = vec![x, x + 1, x - 1];
            assert_eq!(v, msg);
        }
    }

    let mut x = rank;
    world.send_receive_replace_into(&mut x, next_rank, previous_rank);
    assert_eq!(x, previous_rank);
}
