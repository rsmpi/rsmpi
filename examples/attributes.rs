#![deny(warnings)]
use mpi::traits::*;

#[derive(Debug, PartialEq)]
struct MyData1(i32);

impl CommAttribute for MyData1 {
    const CLONE_ON_DUP: bool = true;
}

// These implementations are only for logging
impl Drop for MyData1 {
    fn drop(&mut self) {
        println!("Dropping: {self:?}");
    }
}

impl Clone for MyData1 {
    fn clone(&self) -> Self {
        println!("Cloning: {self:?}");
        Self(self.0 + 1)
    }
}

#[derive(Debug, PartialEq)]
struct MyData2(i32);

impl CommAttribute for MyData2 {}

impl Drop for MyData2 {
    fn drop(&mut self) {
        println!("Dropping: {self:?}");
    }
}

impl Clone for MyData2 {
    fn clone(&self) -> Self {
        println!("Cloning: {self:?}");
        Self(self.0 + 1)
    }
}

fn with_trait_object(comm: &mut dyn Communicator) {
    comm.set_attr(MyData2(22));
    println!("Got from comm2: {:?}", comm.get_attr::<MyData2>());
}

fn main() {
    let universe = mpi::initialize().unwrap();
    println!("Universe: {:?}", universe.size());
    let mut world = universe.world();
    println!("World   : {}", world.size());
    world.set_attr(MyData1(10));
    world.set_attr(MyData2(20));
    assert_eq!(world.get_attr::<MyData1>(), Some(MyData1(10)).as_ref());
    println!("Got from world: {:?}", world.get_attr::<MyData1>());
    assert_eq!(world.get_attr::<MyData2>(), Some(MyData2(20)).as_ref());
    println!("Got from world: {:?}", world.get_attr::<MyData2>());
    let comm = world.duplicate();
    assert_eq!(comm.get_attr::<MyData1>(), Some(MyData1(11)).as_ref());
    println!("Got from comm: {:?}", comm.get_attr::<MyData1>());
    assert_eq!(comm.get_attr::<MyData2>(), None);
    println!("Got from comm: {:?}", comm.get_attr::<MyData2>());
    let mut comm2 = comm.duplicate();
    with_trait_object(&mut comm2);
}
