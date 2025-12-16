#!/bin/bash
# Run tests with flags to avoid sqlite3 dependency issues
# The -p no:cov flag disables the coverage plugin which requires sqlite3

cd "$(dirname "$0")"
python -m pytest --disable-warnings -p no:cov "$@"
