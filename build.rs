use rustc_version::Version as RustcVersion;

fn main() {
    if rustc_version::version().unwrap() >= RustcVersion::parse("1.46.0").unwrap() {
        println!("cargo:rustc-cfg=track_caller_supported");
    }

    let is_msmpi = match build_probe_mpi::probe() {
        Ok(lib) => lib.version == "MS-MPI",
        _ => false,
    };

    if is_msmpi {
        println!("cargo:rustc-cfg=msmpi");
    }
}
