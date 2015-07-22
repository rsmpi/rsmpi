extern crate mpi;

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    println!("Hello parallel world from process {} of {}!", rank, size);

    if size % 2 == 0 {
        match rank % 2 {
            0 => {
                let (msg, status) = world.send_receive::<_, mpi::topology::Rank>(&rank, rank + 1, rank + 1);
                println!("Process {} got message {}.\nStatus is: {:?}", rank, msg, status);
                assert_eq!(msg, rank + 1);
                // FIXME: re-enable if cfg!(feature = "mpi30")
                // let msg = vec![4.0f64, 8.0, 15.0];
                // world.process_at_rank(rank + 1).send(&msg[..]);
            }
            1 => {
                let (msg, status) = world.send_receive::<_, mpi::topology::Rank>(&rank, rank - 1, rank - 1);
                println!("Process {} got message {}.\nStatus is: {:?}", rank, msg, status);
                assert_eq!(msg, rank - 1);
                // FIXME: re-enable if cfg!(feature = "mpi30")
                // let (msg, status) = world.receive_vec::<f64>();
                // println!("Process {} got long message {:?}.\nStatus is: {:?}", rank, msg, status);
            }
            _ => unreachable!()
        }
    } else {
        panic!("Size of MPI_COMM_WORLD must be a multiple of 2, but is {}!", size);
    }
}
