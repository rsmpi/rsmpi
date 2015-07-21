extern crate mpi;

use mpi::traits::*;

use mpi::datatype::{UserDatatype, View};
use mpi::topology::Rank;

trait Modulo<RHS = Self> {
    type Output = Self;
    fn modulo(self, rhs: RHS) -> Self::Output;
}

impl Modulo for Rank {
    type Output = Rank;
    fn modulo(self, rhs: Rank) -> Rank {
        ((self % rhs) + rhs) % rhs
    }
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    let fact = rank as f64;
    let mut b1 = (1..).map(|x| fact * x as f64).take(6).collect::<Vec<_>>();
    let mut b2 = std::iter::repeat(-1.0f64).take(6).collect::<Vec<_>>();
    println!("Rank {} sending message: {:?}.", rank, b1);

    let t = UserDatatype::vector(2, 2, 3, f64::equivalent_datatype());
    let status;
    {
        let mut v1 = unsafe { View::with_count_and_datatype(&mut b1[..], 1, &t) };
        let mut v2 = unsafe { View::with_count_and_datatype(&mut b2[..], 1, &t) };
        status = world.send_receive_into(&mut v1, (rank + 1).modulo(size),
            &mut v2, (rank - 1).modulo(size));
    }

    world.barrier();

    println!("Rank {} received message: {:?}, status: {:?}.", rank, b2, status);
}
