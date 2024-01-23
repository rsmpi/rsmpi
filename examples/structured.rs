#![deny(warnings)]

use mpi::{datatype::UserDatatype, topology::Process, traits::*};
use std::mem::size_of;

struct MyInts([i32; 3]);

unsafe impl Equivalence for MyInts {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1, 1, 1],
            &[
                // Order the logical fields in reverse of their storage order
                (size_of::<i32>() * 2) as mpi::Address,
                size_of::<i32>() as mpi::Address,
                0,
            ],
            &[i32::equivalent_datatype(); 3],
        )
    }
}

fn prepare_on_root(process: Process, ints: &mut [i32]) {
    for i in ints.iter_mut() {
        *i = if process.is_self() { *i + 10 } else { -1 };
    }
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let root_process = world.process_at_rank(0);
    let second_root = world.process_at_rank(1 % world.size());

    if root_process.is_self() {
        let mut ints = MyInts([3, 2, 1]);
        root_process.broadcast_into(&mut ints);
        assert_eq!([3, 2, 1], ints.0);
        prepare_on_root(second_root, &mut ints.0);
        second_root.broadcast_into(&mut ints);
        assert_eq!([13, 12, 11], ints.0);
    } else {
        let mut ints: [i32; 3] = [0, 0, 0];
        root_process.broadcast_into(&mut ints[..]);
        assert_eq!([1, 2, 3], ints);
        prepare_on_root(second_root, &mut ints);
        second_root.broadcast_into(&mut ints);
        assert_eq!([11, 12, 13], ints);
    }
}
