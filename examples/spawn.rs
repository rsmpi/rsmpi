use std::env;
use std::ffi::{CStr, CString};

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    if let Some(parent) = world.parent() {
        assert_eq!("from_parent", env::args().skip(1).next().unwrap());

        parent.process_at_rank(0).send(&7i32);
    } else {
        let exe = CString::new(
            env::current_exe()
                .unwrap()
                .into_os_string()
                .into_string()
                .unwrap(),
        )
        .unwrap();

        let child = world.process_at_rank(0).spawn(
            &exe,
            &[CStr::from_bytes_with_nul(b"from_parent\0").unwrap()],
            1,
        );

        assert_eq!(1, child.remote_size());

        if world.rank() == 0 {
            assert_eq!(7i32, child.process_at_rank(0).receive().0);
        }
    }
}
