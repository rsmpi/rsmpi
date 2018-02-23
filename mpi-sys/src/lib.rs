#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(missing_copy_implementations)]
#![cfg_attr(test, allow(trivial_casts))]
#![cfg_attr(feature="cargo-clippy", allow(used_underscore_binding))]
#![cfg_attr(feature="cargo-clippy", allow(expl_impl_clone_on_copy))]
#![cfg_attr(feature="cargo-clippy", allow(unreadable_literal))]
include!(concat!(env!("OUT_DIR"), "/functions_and_types.rs"));
