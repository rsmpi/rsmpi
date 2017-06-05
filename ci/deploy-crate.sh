#!/bin/sh

set -e

(
  cd build-probe-mpi
  cargo publish || true
)
cargo publish
