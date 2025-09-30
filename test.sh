#!/bin/bash
set -e
echo "Compile..."
python tests/test_compile.py
echo "Run..."
echo "-----------------------------------"
mkdir bin -p
gcc -Wall -Wextra -Wconversion -Wsign-conversion -Wshadow -Wstrict-overflow -Werror -g src/runner/runmem2.c -o bin/runmem2
./bin/runmem2 test.copapy