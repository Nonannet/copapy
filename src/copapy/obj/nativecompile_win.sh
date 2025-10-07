#!/bin/bash
set -e
set -v

SRC=src/copapy/stencils.c
DEST=src/copapy/obj

python src/copapy/generate_stencils.py

mkdir -p $DEST
x86_64-w64-mingw32-gcc -c $SRC -O0 -o $DEST/stencils_AMD64_O0.o
x86_64-w64-mingw32-gcc -c $SRC -O1 -o $DEST/stencils_AMD64_O1.o
x86_64-w64-mingw32-gcc -c $SRC -O2 -o $DEST/stencils_AMD64_O2.o
x86_64-w64-mingw32-gcc -c $SRC -O3 -o $DEST/stencils_AMD64_O3.o

x86_64-w64-mingw32-gcc --version
