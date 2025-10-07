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

SRC=src/copapy/stencils.c
DEST=src/copapy/obj
OPT=O3

mkdir -p $DEST

# Windows x86_64 (ARM64)
python src/copapy/generate_stencils.py --abi ms $SRC
gcc-12 -$OPT -c $SRC -o $DEST/stencils_AMD64_$OPT.o

# Native x86_64
python src/copapy/generate_stencils.py $SRC
gcc-12 -$OPT -c $SRC -o $DEST/stencils_x86_64_$OPT.o

# ARM64
aarch64-linux-gnu-gcc-12 -$OPT -c $SRC -o $DEST/stencils_aarch64_$OPT.o

# ARMv7
arm-linux-gnueabihf-gcc-12 -$OPT -c $SRC -o $DEST/stencils_armv7_$OPT.o

# PowerPC64LE
# powerpc64le-linux-gnu-gcc-12 -$OPT -c $SRC -o $DEST/stencils_ppc64le_$OPT.o

# S390x
# s390x-linux-gnu-gcc-12 -$OPT -c $SRC -o $DEST/stencils_s390x_$OPT.o

# Mips
mips-linux-gnu-gcc-12 -$OPT -c $SRC -o $DEST/stencils_mips_$OPT.o

# RISCV 32 Bit
riscv64-linux-gnu-gcc-12 -$OPT -march=rv32imac -mabi=ilp32 -c $SRC -o $DEST/stencils_riscv32_$OPT.o

# RISCV 64 Bit
riscv64-linux-gnu-gcc-12 -$OPT -c $SRC -o $DEST/stencils_riscv64_$OPT.o