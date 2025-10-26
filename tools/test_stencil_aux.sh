#!/bin/bash

set -e
set -v

mkdir -p bin
FILE=aux_functions
SRC=stencils/$FILE.c
DEST=bin
OPT=O3

mkdir -p $DEST

# Compile native x86_64
gcc -g -$OPT $SRC -o $DEST/$FILE
chmod +x $DEST/$FILE

# Run
$DEST/$FILE