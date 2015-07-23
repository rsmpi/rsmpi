#!/bin/sh

set -e

TOKEN=${GH_TOKEN:?'GH_TOKEN not set!'}
REPO=${TRAVIS_REPO_SLUG:?'TRAVIS_REPO_SLUG not set!'}

echo "Writing OAuth token to .netrc."
touch ${HOME}/.netrc
chmod 0600 ${HOME}/.netrc
echo "machine github.com login $TOKEN password x-oauth-basic" >> ${HOME}/.netrc

echo "Setting origin to use HTTPS for pushing"
git remote set-url --push origin "https://github.com/${REPO}"
