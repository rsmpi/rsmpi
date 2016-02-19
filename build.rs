// Compiles the `rsmpi` C shim library.
extern crate gcc;
// Generates the Rust header for the C API.
extern crate bindgen;

use std::{env, process};

/// splits a command line by space and collects all arguments that start with `prefix`
fn collect_args_with_prefix<'a>(cmd: &'a str, prefix: &str) -> Vec<&'a str> {
    cmd
        .split_whitespace()
        .filter(|arg| arg.starts_with(prefix))
        .collect()
}

fn main() {
    // Use `mpicc` wrapper rather than the system C compiler.
    env::set_var("CC", "mpicc");
    // Build the `rsmpi` C shim library.
    gcc::compile_library("librsmpi.a", &["src/ffi/rsmpi.c"]);

    // Capture the output of `mpicc -show`. This usually gives the actual compiler command line
    // invoked by the `mpicc` compiler wrapper.
    let output = String::from_utf8(
        process::Command::new("mpicc").arg("-show").output().unwrap().stdout).unwrap();
    // Collect the libraries that an MPI C program should be linked to...
    let libs = collect_args_with_prefix(output.as_ref(), "-l");
    // ... and the library search directories...
    let libdirs = collect_args_with_prefix(output.as_ref(), "-L");
    // ... and the header search directories.
    let headerdirs = collect_args_with_prefix(output.as_ref(), "-I");

    // Let `rustc` know about the library search directories.
    for dir in libdirs.iter() { println!("cargo:rustc-link-search=native={}", &dir[2..]); }

    let mut builder = bindgen::builder();
    // Let `bindgen` know about libraries and search directories.
    for lib in libs.iter() { builder.link(&lib[2..]); }
    for dir in libdirs.iter() { builder.clang_arg(*dir); }
    for dir in headerdirs.iter() { builder.clang_arg(*dir); }

    // Tell `bindgen` where to find system headers.
    if let Some(clang_include_dir) = bindgen::get_include_dir() {
        builder.clang_arg("-I");
        builder.clang_arg(clang_include_dir);
    }
    // Generate Rust bindings for the MPI C API.
    let bindings = builder
        .header("src/ffi/rsmpi.h")
        .emit_builtins()
        .generate()
        .unwrap();

    // Write the bindings to disk.
    bindings
        .write_to_file("src/ffi/functions_and_types.rs")
        .unwrap()
}
