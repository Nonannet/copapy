#!/bin/bash
set -e
echo "Compile..."
python tests/test_compile.py
echo "Run..."
echo "-----------------------------------"
./bin/coparun test.copapy