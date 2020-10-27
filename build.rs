fn main() {
    if cfg!(windows) {
        // Adds a cfg to identify MS-MPI. This should perhaps be a more robust check in the future.
        println!("cargo:rustc-cfg=msmpi");
    }
}
