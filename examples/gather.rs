extern crate mpi;

use mpi::traits::*;
use mpi::datatype::{UserDatatype, View, MutView};
use mpi::Count;

fn main () {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let root_rank = 0;
    let root_process = world.process_at_rank(root_rank);

    let count = world.size() as usize;
    let i = 2_u64.pow(world.rank() as u32 + 1);
    let mut a = if world.rank() == root_rank {
        Some(vec![0u64; count])
    } else {
        None
    };

    root_process.gather_into(&i, a.as_mut().map(|x| &mut x[..]));

    if world.rank() == root_rank {
        let a = a.unwrap();
        println!("Root gathered sequence: {:?}.", a);
        assert!(a.iter().enumerate().all(|(a, &b)| b == 2u64.pow(a as u32 + 1)));
    }

    let factor = world.rank() as u64 + 1;
    let a = (1_u64..).take(count).map(|x| x * factor).collect::<Vec<_>>();
    let mut t = if world.rank() == root_rank { Some(vec![0u64; count * count]) } else { None };

    root_process.gather_into(&a[..], t.as_mut().map(|x| &mut x[..]));

    if world.rank() == root_rank {
        let t = t.unwrap();
        println!("Root gathered table:");
        for r in t.chunks(count) {
            println!("{:?}", r);
        }
        assert!((0_u64..).zip(t.iter()).all(|(a, &b)| b == (a / count as u64 + 1) * (a % count as u64 + 1)));
    }

    let d = UserDatatype::contiguous(count as Count, u64::equivalent_datatype());
    t = if world.rank() == root_rank { Some(vec![0u64; count * count]) } else { None };

    {
        let sv = unsafe { View::with_count_and_datatype(&a[..], 1, &d) };
        let mut rv = t.as_mut().map(
            |t| unsafe { MutView::with_count_and_datatype(&mut t[..], count as Count, &d) }
        );

        root_process.gather_into(&sv, rv.as_mut());
    }

    if world.rank() == root_rank {
        let t = t.unwrap();
        println!("Root gathered table:");
        for r in t.chunks(count) {
            println!("{:?}", r);
        }
        assert!((0_u64..).zip(t.iter()).all(|(a, &b)| b == (a / count as u64 + 1) * (a % count as u64 + 1)));
    }
}
