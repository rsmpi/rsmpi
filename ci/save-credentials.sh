#!/bin/sh

set -e

TOKEN=${GH_TOKEN:?'GH_TOKEN not set!'}
REPO=${TRAVIS_REPO_SLUG:?'TRAVIS_REPO_SLUG not set!'}

echo "Writing OAuth token to git credentials"
git config credential.helper store
echo "https://${TOKEN}:@github.com" >> ${HOME}/.git-credentials

echo "Setting origin to use HTTPS for pushing"
git remote set-url origin "https://github.com/${REPO}"
