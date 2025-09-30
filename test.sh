#!/bin/bash
set -e
echo "Compile..."
python tests/test_compile.py
echo "Run..."
echo "-----------------------------------"
gcc -Wall -Wextra -Wconversion -Wsign-conversion -Wshadow -Wstrict-overflow -Werror -g src/runner/runmem2.c -o runmem2
./runmem2 test.copapy