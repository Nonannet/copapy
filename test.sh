#!/bin/bash
set -e
echo "Compile..."
python tests/test_compile.py
echo "Run..."
echo "-----------------------------------"
mkdir bin -p
gcc -Wall -Wextra -Wconversion -Wsign-conversion -Wshadow -Wstrict-overflow -Werror -g src/coparun/runmem.c src/coparun/coparun.c -o bin/coparun
./bin/coparun test.copapy