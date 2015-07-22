#!/bin/sh

set -e

touch ${HOME}/.netrc
chmod 0600 ${HOME}/.netrc
echo "machine github.com login ${GH_TOKEN:?'GH_TOKEN not set!'} password x-oauth-basic" >> ${HOME}/.netrc
