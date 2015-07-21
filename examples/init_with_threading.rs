extern crate mpi;

use mpi::Threading;

fn main() {
    let (universe, threading) = mpi::initialize_with_threading(Threading::Multiple).unwrap();
    assert_eq!(threading, universe.threading_support());
    println!("Supported level of threading: {:?}", threading);
}
