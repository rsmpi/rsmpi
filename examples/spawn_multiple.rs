use std::env;
use std::process::Command;

use mpi::traits::*;

fn main() -> Result<(), mpi::MpiError> {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    if let Some(parent) = world.parent() {
        let child_name = env::args().skip(1).next().unwrap();
        let appnum = universe.appnum().unwrap_or(-1);
        println!(
            "[{}/{}] {} ({}) has parent size {}, universe {:?}",
            world.rank(),
            world.size(),
            child_name,
            appnum,
            parent.remote_size(),
            universe.size(),
        );
    } else {
        fn self_with_arg(arg: &str) -> Command {
            let mut exe = Command::new(env::current_exe().unwrap());
            exe.arg(arg);
            exe
        }
        let child_sizes = [2, 1];
        let commands = vec![self_with_arg("FirstChild"), self_with_arg("SecondChild")];

        let child = world
            .process_at_rank(0)
            .spawn_multiple(&commands, &child_sizes)?;

        assert_eq!(child_sizes.iter().sum::<i32>(), child.remote_size());

        println!(
            "[{}/{}] Parent universe {:?}",
            world.rank(),
            world.size(),
            universe.size(),
        );
    }
    Ok(())
}
