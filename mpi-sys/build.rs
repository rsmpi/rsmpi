// Compiles the `rsmpi` C shim library.
extern crate cc;
// Generates the Rust header for the C API.
extern crate bindgen;
// Finds out information about the MPI library
extern crate build_probe_mpi;

use std::env;
use std::path::Path;

fn main() {
    // Try to find an MPI library
    let lib = match build_probe_mpi::probe() {
        Ok(lib) => lib,
        Err(errs) => {
            println!("Could not find MPI library for various reasons:\n");
            for (i, err) in errs.iter().enumerate() {
                println!("Reason #{}:\n{}\n", i, err);
            }
            panic!();
        }
    };

    // Use `mpicc` wrapper on Unix rather than the system C compiler.
    if cfg!(windows) {
        let mut builder = cc::Build::new();

        builder.file("src/rsmpi.c");

        for inc in &lib.include_paths {
            builder.include(inc);
        }

        builder.compile("librsmpi.a");

        // Adds a cfg to identify MS-MPI
        println!("cargo:rustc-cfg=msmpi");
    } else {
        env::set_var("CC", "mpicc");
        // Build the `rsmpi` C shim library.
        cc::Build::new().file("src/rsmpi.c").compile("librsmpi.a");
    }

    // Let `rustc` know about the library search directories.
    for dir in &lib.lib_paths {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    for lib in &lib.libs {
        println!("cargo:rustc-link-lib={}", lib);
    }

    let mut builder = bindgen::builder();
    // Let `bindgen` know about libraries and search directories.
    for lib in &lib.libs {
        builder = builder.link(lib.clone());
    }
    for dir in &lib.lib_paths {
        builder = builder.clang_arg(format!("-L{}", dir.display()));
    }
    for dir in &lib.include_paths {
        builder = builder.clang_arg(format!("-I{}", dir.display()));
    }

    // Generate Rust bindings for the MPI C API.
    let bindings = builder
        .header("src/rsmpi.h")
        .emit_builtins()
        .blacklist_type("mpich_struct_mpi_long_double_int")
        .blacklist_type("max_align_t") // https://github.com/servo/rust-bindgen/issues/550
        .generate()
        .unwrap();

    // Write the bindings to disk.
    let out_dir = env::var("OUT_DIR").expect("cargo did not set OUT_DIR");
    let out_file = Path::new(&out_dir).join("functions_and_types.rs");
    bindings.write_to_file(out_file).unwrap();
}
