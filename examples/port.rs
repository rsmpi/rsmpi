use std::{
    env,
    io::{Read, Write},
    process::{Command, Stdio},
};

use mpi::{
    collective::{Port, RemotePort},
    topology::MergeOrder,
    traits::*,
};

fn main() -> Result<(), mpi::MpiError> {
    println!("Hi");
    let mut child = if let Ok(_) = env::var("AS_CHILD") {
        None
    } else {
        Some(
            Command::new(env::current_exe().unwrap())
                .arg("--nocapture")
                .env("AS_CHILD", "1")
                .stdin(Stdio::piped())
                .stdout(Stdio::inherit())
                .spawn()
                .expect("failed to spawn child"),
        )
    };

    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let (intercomm, order) = if let Some(ref mut child) = child {
        let port = Port::open();
        let mut stdin = child.stdin.take().unwrap();
        stdin
            .write_all(port.to_string().as_bytes())
            .expect("Parent could not write port for child");
        drop(stdin);
        println!("Parent will accept on {}", port);
        (world.this_process().accept(&port), MergeOrder::Low)
    } else {
        let mut from_parent = std::io::stdin();
        let mut port_name = Vec::new();
        from_parent
            .read_to_end(&mut port_name)
            .expect("Child could not read port on stdin");
        let port_name = String::from_utf8_lossy(&port_name);
        let port = RemotePort::new(&port_name)?;
        println!("Child will connect to {}", port);
        (world.this_process().connect(&port), MergeOrder::High)
    };
    println!("Hi");
    let comm = intercomm.merge(order);
    println!(
        "[{}/{}] Hi, I'm {}",
        comm.rank(),
        comm.size(),
        if child.is_some() { "Parent" } else { "Child" }
    );

    if let Some(mut c) = child {
        let status = c.wait().expect("Child failed to exit");
        assert!(status.success(), "Child exited with an error");
    }
    Ok(())
}
