extern crate mpi;

fn main() {
    let (version, subversion) = mpi::get_version();
    println!("This is MPI-{}.{}.", version, subversion);
    println!("{}", mpi::get_library_version().unwrap());
    let universe = mpi::initialize().unwrap();
    println!("{}", universe.get_processor_name().unwrap());

    assert!(version >= 3, "Rust MPI bindings require MPI standard 3.0 and up.");
}
