use rsmpi::traits::Equivalence;

/// We test if the #[mpi(crate)] macro works correctly.
/// This should compile if the macro correctly uses ``rsmpi`` to address find the mpi crate,
/// because we renamed the dependency.
#[test]
fn derive_custom_path() {
    #[derive(Equivalence)]
    #[mpi(crate = "::rsmpi")]
    struct Particle {
        position: [f64; 3],
    }
}
