#![deny(missing_docs)]
#![warn(missing_copy_implementations)]
#![warn(trivial_casts)]
#![warn(trivial_numeric_casts)]
#![warn(unused_extern_crates)]
#![warn(unused_import_braces)]
#![warn(unused_qualifications)]

use super::super::Library;
use std::{env, error::Error, path::PathBuf};

/// Probe the environment for MS-MPI
pub fn probe() -> Result<Library, Vec<Box<Error>>> {
    let mut errs = vec![];

    let include_path = match env::var("MSMPI_INC") {
        Ok(include_path) => Some(include_path),
        Err(err) => {
            let err: Box<Error> = Box::new(err);
            errs.push(err);
            None
        }
    };

    let lib_env = if cfg!(target_arch = "i686") {
        "MSMPI_LIB32"
    } else if cfg!(target_arch = "x86_64") {
        "MSMPI_LIB64"
    } else {
        panic!("rsmpi does not support your windows architecture!");
    };

    let lib_path = match env::var(lib_env) {
        Ok(lib_path) => Some(lib_path),
        Err(err) => {
            let err: Box<Error> = Box::new(err);
            errs.push(err);
            None
        }
    };

    if errs.len() > 0 {
        return Err(errs);
    }

    Ok(Library {
        libs: vec!["msmpi".to_owned()],
        lib_paths: vec![lib_path.map(PathBuf::from).unwrap()],
        include_paths: vec![include_path.map(PathBuf::from).unwrap()],
        version: String::from("unknown"),
        _priv: (),
    })
}
