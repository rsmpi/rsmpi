#![deny(missing_docs)]
#![warn(missing_copy_implementations)]
#![warn(trivial_casts)]
#![warn(trivial_numeric_casts)]
#![warn(unused_extern_crates)]
#![warn(unused_import_braces)]
#![warn(unused_qualifications)]

use core::fmt;
use std::{self, env, error::Error, path::PathBuf, process::Command};

use super::super::Library;

use pkg_config::Config;

#[derive(Debug, PartialEq)]
struct UnquoteError {
    quote: char,
}

impl UnquoteError {
    fn new(quote: char) -> UnquoteError {
        UnquoteError { quote }
    }
}
impl Error for UnquoteError {}

impl fmt::Display for UnquoteError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Quotes '{}' not closed.", self.quote)
    }
}

fn unquote(s: &str) -> Result<String, UnquoteError> {
    if s.chars().count() < 2 {
        return Ok(String::from(s));
    }

    let quote = s.chars().next().unwrap();

    if quote != '"' && quote != '\'' && quote != '`' {
        return Ok(String::from(s));
    }

    if s.chars().last().unwrap() != quote {
        return Err(UnquoteError::new(quote));
    }

    let s = &s[1..s.len() - 1];
    Ok(String::from(s))
}

#[cfg(test)]
mod tests {
    use super::unquote;

    #[test]
    fn double_quote() {
        let s = "\"/usr/lib/my-mpi/include\"";
        assert_eq!(Ok(String::from("/usr/lib/my-mpi/include")), unquote(s));
    }

    #[test]
    fn single_quote() {
        let s = "'/usr/lib/my-mpi/include'";
        assert_eq!(Ok(String::from("/usr/lib/my-mpi/include")), unquote(s));
    }

    #[test]
    fn backtick_quote() {
        let s = "`/usr/lib/my-mpi/include`";
        assert_eq!(Ok(String::from("/usr/lib/my-mpi/include")), unquote(s));
    }

    #[test]
    fn no_quote() {
        let s = "/usr/lib/my-mpi/include";
        assert_eq!(Ok(String::from("/usr/lib/my-mpi/include")), unquote(s));
    }

    #[test]
    fn unclosed_quote() {
        let s = "'/usr/lib/my-mpi/include";
        assert_eq!(unquote(s).unwrap_err().quote, '\'');
        assert!(unquote(s).is_err());
    }
}

impl From<pkg_config::Library> for Library {
    fn from(lib: pkg_config::Library) -> Self {
        Library {
            mpicc: None,
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
            .filter_map(|x| unquote(&x).ok())
            .map(PathBuf::from)
            .collect();
        // ... and the header search directories.
        let headerdirs = collect_args_with_prefix(output.as_ref(), "-I")
            .into_iter()
            .filter_map(|x| unquote(&x).ok())
            .map(PathBuf::from)
            .collect();

        Library {
            mpicc: Some(mpicc.to_string()),
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
pub fn probe() -> Result<Library, Vec<Box<dyn Error>>> {
    let mut errs = vec![];

    if let Ok(mpi_pkg_config) = env::var("MPI_PKG_CONFIG") {
        match Config::new().cargo_metadata(false).probe(&mpi_pkg_config) {
            Ok(lib) => return Ok(Library::from(lib)),
            Err(err) => {
                let err: Box<dyn Error> = Box::new(err);
                errs.push(err)
            }
        }
    }

    if let Ok(cray_mpich_dir) = env::var("CRAY_MPICH_DIR") {
        let pkg_config_mpich: PathBuf = [&cray_mpich_dir, "lib", "pkgconfig", "mpich.pc"]
            .iter()
            .collect();
        match Config::new()
            .cargo_metadata(false)
            .probe(&pkg_config_mpich.to_string_lossy())
        {
            Ok(lib) => return Ok(Library::from(lib)),
            Err(err) => {
                let err: Box<dyn Error> = Box::new(err);
                errs.push(err)
            }
        }
    }

    match probe_via_mpicc(&env::var("MPICC").unwrap_or_else(|_| String::from("mpicc"))) {
        Ok(lib) => return Ok(lib),
        Err(err) => {
            let err: Box<dyn Error> = Box::new(err);
            errs.push(err)
        }
    }

    match Config::new().cargo_metadata(false).probe("mpich") {
        Ok(lib) => return Ok(Library::from(lib)),
        Err(err) => {
            let err: Box<dyn Error> = Box::new(err);
            errs.push(err)
        }
    }

    match Config::new().cargo_metadata(false).probe("ompi") {
        Ok(lib) => return Ok(Library::from(lib)),
        Err(err) => {
            let err: Box<dyn Error> = Box::new(err);
            errs.push(err)
        }
    }

    Err(errs)
}
