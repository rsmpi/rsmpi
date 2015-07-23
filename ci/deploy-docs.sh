#!/bin/sh

set -e

travis-cargo doc

# Manual doc deployment
echo "<meta http-equiv=refresh content=0;url=mpi/index.html>" >> target/doc/index.html
ghp-import -n target/doc
git remote -v
git push -v origin gh-pages
