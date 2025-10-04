#!/bin/bash

#Setup:
#sudo apt-get update
#sudo apt-get install -y \
#gcc-12-aarch64-linux-gnu \
#gcc-12-arm-linux-gnueabihf \
#gcc-12-powerpc64le-linux-gnu \
#gcc-12-s390x-linux-gnu \
#gcc-12-mips-linux-gnu \
#gcc-12-riscv64-linux-gnu

set -e
set -v
python src/copapy/generate_stencils.py
SRC=src/copapy/stencils.c
DEST=src/copapy/obj
mkdir -p $DEST

# Native x86_64
gcc-12 -c $SRC -o $DEST/stencils_x86_64.o

# ARM64
aarch64-linux-gnu-gcc-12 -O3 -c $SRC -o $DEST/stencils_aarch64.o

# ARMv7
arm-linux-gnueabihf-gcc-12 -O3 -c $SRC -o $DEST/stencils_armv7.o

# PowerPC64LE
# powerpc64le-linux-gnu-gcc-12 -O3 -c $SRC -o $DEST/stencils_ppc64le.o

# S390x
# s390x-linux-gnu-gcc-12 -O3 -c $SRC -o $DEST/stencils_s390x.o

# Mips
mips-linux-gnu-gcc-12 -O3 -c $SRC -o $DEST/stencils_mips.o

# RISCV 32 Bit
riscv64-linux-gnu-gcc-12 -O3 -march=rv32imac -mabi=ilp32 -c $SRC -o $DEST/stencils_riscv32.o

# RISCV 64 Bit
riscv64-linux-gnu-gcc-12 -O3 -c $SRC -o $DEST/stencils_riscv64.o