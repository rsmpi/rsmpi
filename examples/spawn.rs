use std::env;
use std::process::Command;

use mpi::topology::MergeOrder;
use mpi::traits::*;

fn main() -> Result<(), mpi::MpiError> {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let merged = if let Some(parent) = world.parent() {
        assert_eq!("from_parent", env::args().skip(1).next().unwrap());

        parent.process_at_rank(0).send(&7i32);

        println!(
            "[{}/{}] Child universe {:?}",
            world.rank(),
            world.size(),
            universe.size()
        );
        parent.merge(MergeOrder::High)
    } else {
        let child_size = 1;
        let mut exe = Command::new(env::current_exe().unwrap());
        exe.arg("from_parent");

        let child = world.process_at_rank(0).spawn(&exe, child_size)?;

        assert_eq!(child_size, child.remote_size());

        if world.rank() == 0 {
            assert_eq!(7i32, child.process_at_rank(0).receive().0);
        }
        println!(
            "[{}/{}] Parent universe {:?}",
            world.rank(),
            world.size(),
            universe.size(),
        );
        child.merge(MergeOrder::Low)
    };
    println!(
        "[{}/{}] Merged is World {}/{}",
        merged.rank(),
        merged.size(),
        world.rank(),
        world.size()
    );
    Ok(())
}
