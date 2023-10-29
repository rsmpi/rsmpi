#![deny(warnings)]

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    println!("Universe: {:?}", universe.size());
    println!("World   : {}", universe.world().size());
}
