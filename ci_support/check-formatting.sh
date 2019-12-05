#!/usr/bin/env bash

# Handy script to check the formatting of all files that are different between
# the current git commit and the head commit on master.

CURRENT_COMMIT_HASH=$(git rev-parse HEAD)
MERGE_COMMIT_HASH=$(git rev-parse master)
FORMATTING_ERRORS=$(git clang-format ${MERGE_COMMIT_HASH} ${CURRENT_COMMIT_HASH} --diff)
if [ -z "${FORMATTING_ERRORS}" ]; then
  echo "All modified source code properly formatted!"
  exit 0
else
  echo "Formatting errors in modified source code!"
  echo "${FORMATTING_ERRORS}"
  exit 1
fi
