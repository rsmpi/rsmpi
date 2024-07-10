fn main() {
    // https://blog.rust-lang.org/2024/05/06/check-cfg.html#buildrs-example
    println!("cargo:rustc-check-cfg=cfg(msmpi)");

    let is_msmpi = match build_probe_mpi::probe() {
        Ok(lib) => lib.version == "MS-MPI",
        _ => false,
    };

    if is_msmpi {
        println!("cargo:rustc-cfg=msmpi");
    }
}
