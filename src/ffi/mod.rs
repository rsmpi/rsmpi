// #[c_import(h = "../rsmpi.h", MPI_COMM_WORLD)]
// mod constants { }
// pub use self::constants::*;

mod functions_and_types;
pub use self::functions_and_types::*;

#[cfg(extern_statics_are_unsafe)]
macro_rules! unsafe_extern_static {
    ( $x:expr ) => { unsafe { $x } }
}

#[cfg(not(extern_statics_are_unsafe))]
macro_rules! unsafe_extern_static {
    ( $x:expr ) => { $x }
}
