use rsmpi::traits::Equivalence;

// Attention: If this test case needs to be modified,
// please check if the code examples in mpi-derive/src/lib.rs and for the Equivalence trait are still
// valid. They cannot be checked automatically.

/// We test if the #[mpi(crate = "..")] macro works correctly.
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
