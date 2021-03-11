#![cfg(feature = "derive")]

use mpi::traits::Equivalence;
const CONSTANT: usize = 7;

/// We test that #[derive(Equivalence)] correctly casts CONSTANT to a i32 for the
/// C interop. For defining a rust array, CONSTANT must be usize.
#[test]
fn derive_equivalence() {
    #[derive(Equivalence)]
    struct ArrayWrapper {
        field: [f32; CONSTANT],
    }
}
