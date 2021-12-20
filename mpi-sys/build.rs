// Compiles the `rsmpi` C shim library.
extern crate cc;
// Generates the Rust header for the C API.
extern crate bindgen;
// Finds out information about the MPI library
extern crate build_probe_mpi;

use std::env;
use std::path::Path;
use std::path::PathBuf;
use std::collections::HashMap;

fn main() {

    let mut builder = cc::Build::new();
    builder.file("src/rsmpi.c");

    let include_paths = vec![
        PathBuf::from("/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi"),
        PathBuf::from("/usr/lib/x86_64-linux-gnu/openmpi/include"),
    ];

    let lib_paths = vec![
        PathBuf::from("/usr/lib/x86_64-linux-gnu/openmpi/lib")
    ];

    let libs = vec![
        "mpi"
    ];

    for inc in &include_paths {
        builder.include(inc);
    }

    // builder.compiler("cc");
    builder.compiler("mpicc");

    let compiler = builder.try_get_compiler();

    // Build the `rsmpi` C shim library.\
    builder.compile("rsmpi");

    // Let `rustc` know about the library search directories.
    for dir in &lib_paths {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    for lib in &libs {
        println!("cargo:rustc-link-lib={}", lib);
    }

    let mut builder = bindgen::builder();
    // Let `bindgen` know about header search directories.
    for dir in &include_paths {
        builder = builder.clang_arg(format!("-I{}", dir.display()));
    }

    // Get the same system includes as used to build the "rsmpi" lib. This block only really does
    // anything when targeting msvc.
    if let Ok(compiler) = compiler {
        let include_env = compiler.env().iter().find(|(key, _)| key == "INCLUDE");
        if let Some((_, include_paths)) = include_env {
            if let Some(include_paths) = include_paths.to_str() {
                // Add include paths via -I
                builder = builder.clang_args(include_paths.split(';').map(|i| format!("-I{}", i)));
            }
        }
    }

    // Generate Rust bindings for the MPI C API.
    let bindings = builder
        .header("src/rsmpi.h")
        .emit_builtins()
        .generate()
        .unwrap();

    // Write the bindings to disk.
    let out_dir = env::var("OUT_DIR").expect("cargo did not set OUT_DIR");
    let out_file = Path::new(&out_dir).join("functions_and_types.rs");
    bindings.write_to_file(out_file).unwrap();
}
