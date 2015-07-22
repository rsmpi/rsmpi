#!/bin/sh

set -e

travis-cargo doc
travis-cargo doc-upload
