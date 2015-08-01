extern crate gcc;
extern crate bindgen;
extern crate probe_c_api;

use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::process::Command;

use probe_c_api::Probe;

fn collect_args_with_prefix<'a>(cmd: &'a str, prefix: &str) -> Vec<&'a str> {
    cmd
        .split_whitespace()
        .filter(|arg| arg.starts_with(prefix))
        .collect()
}

fn main() {
    env::set_var("CC", "mpicc");
    gcc::compile_library("librsmpi.a", &["src/rsmpi.c"]);

    let output = String::from_utf8(
        Command::new("mpicc").arg("-show").output().unwrap().stdout).unwrap();
    let libs = collect_args_with_prefix(output.as_ref(), "-l");
    let libdirs = collect_args_with_prefix(output.as_ref(), "-L");
    let headerdirs = collect_args_with_prefix(output.as_ref(), "-I");

    for dir in libdirs.iter() { println!("cargo:rustc-link-search=native={}", &dir[2..]); }

    let mut builder = bindgen::builder();
    for lib in libs.iter() { builder.link(&lib[2..]); }
    for dir in libdirs.iter() { builder.clang_arg(*dir); }
    for dir in headerdirs.iter() { builder.clang_arg(*dir); }
    if let Some(clang_include_dir) = bindgen::get_include_dir() {
        builder.clang_arg("-I");
        builder.clang_arg(clang_include_dir);
    }
    let bindings = builder
        .header("src/rsmpi.h")
        .emit_builtins()
        .generate()
        .unwrap();

    bindings
        .write_to_file("src/ffi/functions_and_types.rs")
        .unwrap();

    let probe = Probe::new(
        vec!["\"mpi.h\"".into()],
        &env::temp_dir(),
        |source_path, exe_path| {
            Command::new("mpicc")
                .arg(source_path)
                .arg("-o").arg(exe_path)
                .output()
        },
        |exe_path| {
            Command::new(exe_path)
                .output()
        }).unwrap();

    let constants = vec![
        ("MPI_UNDEFINED", "int"),

        ("MPI_ANY_SOURCE", "int"),
        ("MPI_ANY_TAG", "int"),

        ("MPI_IDENT", "int"),
        ("MPI_CONGRUENT", "int"),
        ("MPI_SIMILAR", "int"),
        ("MPI_UNEQUAL", "int"),

        ("MPI_THREAD_SINGLE", "int"),
        ("MPI_THREAD_FUNNELED", "int"),
        ("MPI_THREAD_SERIALIZED", "int"),
        ("MPI_THREAD_MULTIPLE", "int"),
    ];

    let mut binding_file = File::create("src/ffi/constants.rs").unwrap();
    writeln!(&mut binding_file, "use ::libc::c_int;").unwrap();

    for (name, typ) in constants {
        let rust_type = if typ == "int" { "c_int" } else { typ };
        if probe.is_signed(typ).unwrap() {
            let val = probe.signed_integer_constant(name).unwrap();
            writeln!(&mut binding_file, "pub const {n}: {t} = {v};", n = name, t = rust_type, v = val).unwrap();
        } else {
            let val = probe.unsigned_integer_constant(name).unwrap();
            writeln!(&mut binding_file, "pub const {n}: {t} = {v};", n = name, t = rust_type, v = val).unwrap();
        }
    }
}
