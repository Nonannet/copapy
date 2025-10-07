#!/bin/bash
set -e
set -v

SRC=src/copapy/stencils.c
DEST=src/copapy/obj

python src/copapy/generate_stencils.py $SRC

mkdir -p $DEST
gcc-12 -c $SRC -O0 -o $DEST/stencils_x86_64_O0.o
gcc-12 -c $SRC -O1 -o $DEST/stencils_x86_64_O1.o
gcc-12 -c $SRC -O2 -o $DEST/stencils_x86_64_O2.o
gcc-12 -c $SRC -O3 -o $DEST/stencils_x86_64_O3.o

python src/copapy/generate_stencils.py --abi ms $SRC

gcc-12 -c $SRC -O0 -o $DEST/stencils_AMD64_O0.o
gcc-12 -c $SRC -O1 -o $DEST/stencils_AMD64_O1.o
gcc-12 -c $SRC -O2 -o $DEST/stencils_AMD64_O2.o
gcc-12 -c $SRC -O3 -o $DEST/stencils_AMD64_O3.o