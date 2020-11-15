fn main() {
    let is_msmpi = build_probe_mpi::probe().map_or(false, |lib| lib.version == "MS-MPI");

    if is_msmpi {
        println!("cargo:rustc-cfg=msmpi");
    }
}
