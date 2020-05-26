#!/usr/bin/env bash

echo "Building ops ..."
rm -rf build
rm -f fairmot/ops/_ext.*
python setup.py build_ext --inplace
rm -rf build
