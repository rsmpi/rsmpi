#![deny(missing_docs)]
#![warn(missing_copy_implementations)]
#![warn(trivial_casts)]
#![warn(trivial_numeric_casts)]
#![warn(unused_extern_crates)]
#![warn(unused_import_braces)]
#![warn(unused_qualifications)]
#![deny(warnings)]

//! Probe an environment for an installed MPI library
//!
//! Probing is done in several steps:
//!
//! 1. Try to find an MPI compiler wrapper either from the environment variable `MPICC` or under
//!    the name `mpicc` then run the compiler wrapper with the command line argument `-show` and
//!    interpret the resulting output as `gcc` compatible command line arguments.
//! 2. Query the `pkg-config` database for an installation of `mpich`.
//! 3. Query the `pkg-config` database for an installation of `openmpi`.
//!
//! The result of the first successful step is returned. If no step is successful, a list of errors
//! encountered while executing the steps is returned.

extern crate pkg_config;

use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::process::Command;

use pkg_config::Config;

/// Result of a successfull probe
#[derive(Clone, Debug)]
pub struct Library {
    /// Names of the native MPI libraries that need to be linked
    pub libs: Vec<String>,
    /// Search path for native MPI libraries
    pub lib_paths: Vec<PathBuf>,
    /// Search path for C header files
    pub include_paths: Vec<PathBuf>,
    /// The version of the MPI library
    pub version: String,
    _priv: (),
}

impl From<pkg_config::Library> for Library {
    fn from(lib: pkg_config::Library) -> Self {
        Library {
            libs: lib.libs,
            lib_paths: lib.link_paths,
            include_paths: lib.include_paths,
            version: lib.version,
            _priv: (),
        }
    }
}

fn probe_via_mpicc(mpicc: &str) -> std::io::Result<Library> {
    // Capture the output of `mpicc -show`. This usually gives the actual compiler command line
    // invoked by the `mpicc` compiler wrapper.
    Command::new(mpicc).arg("-show").output().map(|cmd| {
        let output = String::from_utf8(cmd.stdout).expect("mpicc output is not valid UTF-8");
        // Collect the libraries that an MPI C program should be linked to...
        let libs = collect_args_with_prefix(output.as_ref(), "-l");
        // ... and the library search directories...
        let libdirs = collect_args_with_prefix(output.as_ref(), "-L")
            .into_iter()
            .map(PathBuf::from)
            .collect();
        // ... and the header search directories.
        let headerdirs = collect_args_with_prefix(output.as_ref(), "-I")
            .into_iter()
            .map(PathBuf::from)
            .collect();

        Library {
            libs,
            lib_paths: libdirs,
            include_paths: headerdirs,
            version: String::from("unknown"),
            _priv: (),
        }
    })
}

/// splits a command line by space and collects all arguments that start with `prefix`
fn collect_args_with_prefix(cmd: &str, prefix: &str) -> Vec<String> {
    cmd.split_whitespace()
        .filter_map(|arg| {
            if arg.starts_with(prefix) {
                Some(arg[2..].to_owned())
            } else {
                None
            }
        })
        .collect()
}

/// Probe the environment for an installed MPI library
pub fn probe() -> Result<Library, Vec<Box<Error>>> {
    let mut errs = vec![];

    match probe_via_mpicc(&env::var("MPICC").unwrap_or_else(|_| String::from("mpicc"))) {
        Ok(lib) => return Ok(lib),
        Err(err) => {
            let err: Box<Error> = Box::new(err);
            errs.push(err)
        }
    }

    match Config::new().cargo_metadata(false).probe("mpich") {
        Ok(lib) => return Ok(Library::from(lib)),
        Err(err) => {
            let err: Box<Error> = Box::new(err);
            errs.push(err)
        }
    }

    match Config::new().cargo_metadata(false).probe("openmpi") {
        Ok(lib) => return Ok(Library::from(lib)),
        Err(err) => {
            let err: Box<Error> = Box::new(err);
            errs.push(err)
        }
    }

    Err(errs)
}
