#!/bin/sh

set -e

(
  cd build-probe-mpi
  cargo publish || true
)
(
  cd mpi-sys
  cargo publish || true
)
cargo publish
