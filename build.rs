extern crate rustc_version;
use rustc_version::Version as RustcVersion;

fn main() {
    // Access to extern statics has to be marked unsafe after 1.13.0
    if rustc_version::version().unwrap() >= RustcVersion::parse("1.13.0").unwrap() {
        println!("cargo:rustc-cfg=extern_statics_are_unsafe");
    }

    if cfg!(windows) {
        // Adds a cfg to identify MS-MPI. This should perhaps be a more robust check in the future.
        println!("cargo:rustc-cfg=msmpi");
    }
}
