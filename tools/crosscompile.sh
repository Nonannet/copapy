#!/bin/bash

set -e
set -v

mkdir -p bin
SRC=bin/stencils.c
DEST=src/copapy/obj
OPT=O3

mkdir -p $DEST

# Windows x86_64 (ARM64)
python tools/generate_stencils.py --abi ms $SRC
gcc-13 -$OPT -c $SRC -o $DEST/stencils_AMD64_$OPT.o

# Windows x86
gcc-13 -m32 -$OPT -c $SRC -o $DEST/stencils_x86_$OPT.o


# Native x86_64
python tools/generate_stencils.py $SRC
gcc-13 -$OPT -c $SRC -o $DEST/stencils_x86_64_$OPT.o

# Native i686
python tools/generate_stencils.py $SRC
gcc-13 -m32 -$OPT -c $SRC -o $DEST/stencils_i686_$OPT.o

# ARM64 linux (aarch64)
aarch64-linux-gnu-gcc-13 -$OPT -c $SRC -o $DEST/stencils_aarch64_$OPT.o

# ARM64 macos (copy aarch64)
cp $DEST/stencils_aarch64_$OPT.o $DEST/stencils_arm64_$OPT.o

# ARMv7
arm-linux-gnueabihf-gcc-13 -$OPT -c $SRC -o $DEST/stencils_armv7_$OPT.o

# PowerPC64LE
# powerpc64le-linux-gnu-gcc-13 -$OPT -c $SRC -o $DEST/stencils_ppc64le_$OPT.o

# S390x
# s390x-linux-gnu-gcc-13 -$OPT -c $SRC -o $DEST/stencils_s390x_$OPT.o

# Mips (Big Endian)
mips-linux-gnu-gcc-13 -$OPT -c $SRC -o $DEST/stencils_mips_$OPT.o

# Mips (Little Endian)
mipsel-linux-gnu-gcc-13 -$OPT -c $SRC -o $DEST/stencils_mips_$OPT.o

# RISCV 32 Bit
riscv64-linux-gnu-gcc-13 -$OPT -march=rv32imac -mabi=ilp32 -c $SRC -o $DEST/stencils_riscv32_$OPT.o

# RISCV 64 Bit
riscv64-linux-gnu-gcc-13 -$OPT -c $SRC -o $DEST/stencils_riscv64_$OPT.o
