#![deny(warnings)]
#![allow(forgetting_copy_types)]
extern crate mpi;
#[cfg(feature = "complex")]
use mpi::traits::*;
#[cfg(feature = "complex")]
use num_complex::Complex64;

fn main() {
    #[cfg(feature = "complex")]
    {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();

        let root_process = world.process_at_rank(0);

        let mut data = if world.rank() == 0 {
            vec![
                Complex64::new(1.0, 2.0),
                Complex64::new(2.0, 3.0),
                Complex64::new(3.0, 4.0),
            ]
        } else {
            Vec::<Complex64>::with_capacity(3)
        };

        root_process.broadcast_into(&mut data);

        assert_eq!(
            vec![
                Complex64::new(1.0, 2.0),
                Complex64::new(2.0, 3.0),
                Complex64::new(3.0, 4.0),
            ],
            data
        );
    }
}
