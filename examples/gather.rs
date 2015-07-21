extern crate mpi;

use mpi::traits::*;

fn main () {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let root_rank = 0;
    let root_process = world.process_at_rank(root_rank);

    let i = 2_u64.pow(world.rank() as u32 + 1);
    let mut a = if world.rank() == root_rank {
        Some(std::iter::repeat(0).take(world.size() as usize).collect::<Vec<u64>>())
    } else {
        None
    };

    root_process.gather_into(&i, a.as_mut().map(|x| &mut x[..]));

    if world.rank() == root_rank {
        let a = a.unwrap();
        println!("Root gathered sequence: {:?}.", a);
        assert!(a.iter().enumerate().all(|(a, &b)| b == 2u64.pow(a as u32 + 1)));
    }

    let count = world.size() as usize;
    let factor = world.rank() as u64 + 1;
    let a = (1_u64..).take(count).map(|x| x * factor).collect::<Vec<_>>();
    let mut t = if world.rank() == root_rank {
        Some(std::iter::repeat(0).take(count * count).collect::<Vec<u64>>())
    } else {
        None
    };

    root_process.gather_into(&a[..], t.as_mut().map(|x| &mut x[..]));

    if world.rank() == root_rank {
        let t = t.unwrap();
        println!("Root gathered table:");
        for r in t.chunks(count) {
            println!("{:?}", r);
        }
        assert!((0_u64..).zip(t.iter()).all(|(a, &b)| b == (a / count as u64 + 1) * (a % count as u64 + 1)));
    }
}
