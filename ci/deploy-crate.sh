#!/bin/sh

set -e

(
  cd build-probe-mpi
  cargo publish --dry-run || true
)
cargo publish --dry-run --allow-dirty
