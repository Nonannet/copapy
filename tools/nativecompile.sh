#!/bin/bash
set -e
set -v

mkdir -p bin
SRC=build/stencils/stencils.c
DEST=src/copapy/obj
FLAGS="-fno-pic"

python stencils/generate_stencils.py $SRC

mkdir -p $DEST
gcc-12 $FLAGS -c $SRC -O0 -o $DEST/stencils_x86_64_O0.o
gcc-12 $FLAGS -c $SRC -O1 -o $DEST/stencils_x86_64_O1.o
gcc-12 $FLAGS -c $SRC -O2 -o $DEST/stencils_x86_64_O2.o
gcc-12 $FLAGS -c $SRC -O3 -o $DEST/stencils_x86_64_O3.o
