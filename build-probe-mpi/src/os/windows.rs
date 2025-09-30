#![deny(missing_docs)]
#![warn(missing_copy_implementations)]
#![warn(trivial_casts)]
#![warn(trivial_numeric_casts)]
#![warn(unused_extern_crates)]
#![warn(unused_import_braces)]
#![warn(unused_qualifications)]

use std::{env, error::Error, fmt, path::PathBuf};

use super::super::Library;

#[derive(Debug)]
struct VarError {
    key: String,
    origin: env::VarError,
}

impl fmt::Display for VarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.key, self.origin)
    }
}

impl Error for VarError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.origin)
    }
}

fn var(key: &str) -> Result<String, VarError> {
    env::var(key).map_err(|e| VarError {
        key: key.to_owned(),
        origin: e,
    })
}

/// Probe the environment for MS-MPI
pub fn probe() -> Result<Library, Vec<Box<dyn Error>>> {
    let mut intel_errs: Vec<Box<dyn Error>> = vec![];

    // Check for Intel MPI first
    match var("I_MPI_ROOT") {
        Ok(oneapi_root) => {
            let include_path = PathBuf::from(&oneapi_root).join("include");

            let lib_path = if PathBuf::from(&oneapi_root).join("lib/release").exists() {
                PathBuf::from(&oneapi_root).join("lib/release")
            } else {
                PathBuf::from(&oneapi_root).join("lib")
            };

            return Ok(Library {
                mpicc: None,
                libs: vec!["impi".to_owned()],
                lib_paths: vec![lib_path],
                include_paths: vec![include_path],
                version: String::from("Intel MPI"),
                _priv: (),
            });
        }
        Err(err) => {
            intel_errs.push(Box::new(err));
        }
    }

    let mut msmpi_errs: Vec<Box<dyn Error>> = vec![];

    let include_path = match var("MSMPI_INC") {
        Ok(include_path) => Some(include_path),
        Err(err) => {
            msmpi_errs.push(Box::new(err));
            None
        }
    };

    let lib_env = if env::var("CARGO_CFG_TARGET_ARCH") == Ok("x86".to_string()) {
        "MSMPI_LIB32"
    } else if env::var("CARGO_CFG_TARGET_ARCH") == Ok("x86_64".to_string()) {
        "MSMPI_LIB64"
    } else {
        panic!("rsmpi does not support your windows architecture!");
    };

    let lib_path = match var(lib_env) {
        Ok(lib_path) => Some(lib_path),
        Err(err) => {
            msmpi_errs.push(Box::new(err));
            None
        }
    };

    if msmpi_errs.len() > 0 && intel_errs.len() > 0 {
        let mut all_errs = msmpi_errs;
        all_errs.extend(intel_errs);
        return Err(all_errs);
    }

    Ok(Library {
        mpicc: None,
        libs: vec!["msmpi".to_owned()],
        lib_paths: vec![lib_path.map(PathBuf::from).unwrap()],
        include_paths: vec![include_path.map(PathBuf::from).unwrap()],
        version: String::from("MS-MPI"),
        _priv: (),
    })
}
