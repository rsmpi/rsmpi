#!/bin/sh

set -e

TOKEN=${GH_TOKEN:?'GH_TOKEN not set!'}

touch ${HOME}/.netrc
chmod 0600 ${HOME}/.netrc
echo "Writing OAuth token to .netrc."
echo "machine github.com login $TOKEN password x-oauth-basic" >> ${HOME}/.netrc

echo "HOME looks like:"
ls -la ${HOME}

echo ".netrc contains:"
grep -o -E "^machine[[:space:]]+[^[:space:]]+[[:space:]]+login" ~/.netrc 

echo "trying clone:"
git clone --verbose https://github.com/davisp/ghp-import
