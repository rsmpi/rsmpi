// #[c_import(h = "../rsmpi.h", MPI_COMM_WORLD)]
// mod constants { }
// pub use self::constants::*;
pub mod constants;

mod functions_and_types;
pub use self::functions_and_types::*;
