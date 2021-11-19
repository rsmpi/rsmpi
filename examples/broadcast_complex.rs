/// Broadcast complex valued data
extern crate mpi;
use mpi::traits::*;
use num_complex::Complex;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let root_process = world.process_at_rank(0);

    let mut data = if world.rank() == 0 {
        [
            Complex::<f64>::new(1., -2.),
            Complex::<f64>::new(8., -4.),
            Complex::<f64>::new(3., -9.),
            Complex::<f64>::new(7., -5.),
        ]
    } else {
        [Complex::<f64>::new(0., 0.); 4]
    };

    // root_process.broadcast_into(&mut data);
    root_process.broadcast_into(&mut data[..]);

    assert_eq!(
        data,
        [
            Complex::<f64>::new(1., -2.),
            Complex::<f64>::new(8., -4.),
            Complex::<f64>::new(3., -9.),
            Complex::<f64>::new(7., -5.),
        ]
    );
}
