#!/bin/bash
set -e
set -v
mkdir -p bin
SRC=bin/stencils.c
DEST=src/copapy/obj
python tools/generate_stencils.py $SRC
mkdir -p $DEST
gcc -c $SRC -O0 -o $DEST/stencils_x86_64_O0.o
gcc -c $SRC -O1 -o $DEST/stencils_x86_64_O1.o
gcc -c $SRC -O2 -o $DEST/stencils_x86_64_O2.o
gcc -c $SRC -O3 -o $DEST/stencils_x86_64_O3.o

mkdir bin -p
gcc -Wall -Wextra -Wconversion -Wsign-conversion -Wshadow -Wstrict-overflow -Werror -g -O3 src/coparun/runmem.c src/coparun/coparun.c src/coparun/mem_man.c -o bin/coparun
#x86_64-w64-mingw32-gcc -Wall -Wextra -Wconversion -Wsign-conversion -Wshadow -Wstrict-overflow -Werror src/runner/runmem2.c -Wall -O3 -o bin/runmem2.exe
