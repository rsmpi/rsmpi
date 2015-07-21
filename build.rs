extern crate gcc;
extern crate bindgen;

use std::{env, process};

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
        process::Command::new("mpicc").arg("-show").output().unwrap().stdout).unwrap();
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
        .unwrap()
}
