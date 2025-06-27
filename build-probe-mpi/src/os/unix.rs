#![deny(missing_docs)]
#![warn(missing_copy_implementations)]
#![warn(trivial_casts)]
#![warn(trivial_numeric_casts)]
#![warn(unused_extern_crates)]
#![warn(unused_import_braces)]
#![warn(unused_qualifications)]

use core::fmt;
use std::{env, error::Error, path::PathBuf, process::Command};

use pkg_config::Config;

use super::super::Library;

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
    shell_words::split(cmd)
        .unwrap()
        .iter()
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
            Err(_err) => {
                let err: Box<dyn Error> = "Error: Environmental variable $MPI_PKG_CONFIG is set, but is not valid. Please check that it holds the address to a file ending in `.pc`.\nSee https://github.com/rsmpi/rsmpi/blob/main/README.md for more details.\nIf you mean to use another method to find an MPI library, unset $MPI_PKG_CONFIG".into();
                errs.push(err);
                return Err(errs);
            }
        }
    } else {
        let err: Box<dyn Error> =
            "Environmental variable $MPI_PKG_CONFIG is not set. Trying next method".into();
        errs.push(err);
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
    } else {
        let err: Box<dyn Error> =
            "Environmental variable $CRAY_MPICH_DIR is not set. Trying next method".into();
        errs.push(err);
    }

    match probe_via_mpicc(&env::var("MPICC").unwrap_or_else(|_| String::from("mpicc"))) {
        Ok(lib) => return Ok(lib),
        Err(err) => {
            let err: Box<dyn Error> =
                format!("MPICC failed with error: {}. Trying next method", err).into();
            errs.push(err);
        }
    }

    match Config::new().cargo_metadata(false).probe("mpich") {
        Ok(lib) => return Ok(Library::from(lib)),
        Err(_err) => {
            let err: Box<dyn Error> =
                "Fallback 1: mpich was not found. To use this fallback, set $PKG_CONFIG_PATH to a path to a folder containing mpich.pc. Note that the recommended installation is through the $MPI_PKG_CONFIG. Trying next method".into();
            errs.push(err);
        }
    }

    match Config::new().cargo_metadata(false).probe("ompi") {
        Ok(lib) => return Ok(Library::from(lib)),
        Err(_err) => {
            let err: Box<dyn Error> =
                "Fallback 2: ompi (Open MPI) was not found. To use this fallback, set $PKG_CONFIG_PATH to a path to a folder containing ompi.pc. Note that the recommended installation is through the $MPI_PKG_CONFIG".into();
            errs.push(err);
        }
    }

    Err(errs)
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

    use super::collect_args_with_prefix;

    #[test]
    fn flag_parsing_with_space() {
        let cmd = r#"gcc -I"/opt/intel/My Oneapi/mpi/2021.8.0/include" -L"/opt/intel/My Oneapi/mpi/2021.8.0/lib/release" -L"/opt/intel/My Oneapi/mpi/2021.8.0/lib" -Xlinker --enable-new-dtags -Xlinker -rpath -Xlinker "/opt/intel/My Oneapi/mpi/2021.8.0/lib/release" -Xlinker -rpath -Xlinker "/opt/intel/My Oneapi/mpi/2021.8.0/lib" -lmpifort -lmpi -lrt -lpthread -Wl,-z,now -Wl,-z,relro -Wl,-z,noexecstack -Xlinker --enable-new-dtags -ldl"#;
        assert_eq!(
            collect_args_with_prefix(cmd, "-L"),
            vec![
                "/opt/intel/My Oneapi/mpi/2021.8.0/lib/release",
                "/opt/intel/My Oneapi/mpi/2021.8.0/lib"
            ]
        );
    }
    #[test]
    fn flag_parsing_without_space() {
        let cmd = r#"gcc -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi"#;
        assert_eq!(
            collect_args_with_prefix(cmd, "-L"),
            vec!["/usr/lib/x86_64-linux-gnu/openmpi/lib"]
        );
    }
}
