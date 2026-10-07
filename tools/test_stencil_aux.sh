#!/bin/bash

set -e
set -v

FILE=test
SRC="stencils/$FILE.c"
DEST=bin
OPT=O3

mkdir -p $DEST

# Compile native x86_64, the math functions used by the aux functions are in libm
gcc -g -$OPT $SRC -o $DEST/$FILE -lm
chmod +x $DEST/$FILE

# Run
$DEST/$FILE
