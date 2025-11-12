#!/bin/bash

set -e
set -v

mkdir -p build/stencils
SRC=build/stencils/stencils.c
DEST=src/copapy/obj
OPT=O3
FLAGS="-fno-pic"

mkdir -p $DEST

# -------------- Compile stencils --------------

python3 stencils/generate_stencils.py $SRC

# x86_64
gcc-13 $FLAGS -$OPT -c $SRC -o $DEST/stencils_x86_64_$OPT.o

# x86 - 32 bit
gcc-13 $FLAGS -m32 -$OPT -c $SRC -o $DEST/stencils_x86_$OPT.o

# ARM64 linux (aarch64)
aarch64-linux-gnu-gcc-13 $FLAGS -$OPT -c $SRC -o $DEST/stencils_arm64_$OPT.o

# ARMv7
arm-linux-gnueabihf-gcc-13 $FLAGS -$OPT -c $SRC -o $DEST/stencils_arm_$OPT.o

# PowerPC64LE
# powerpc64le-linux-gnu-gcc-13 $FLAGS -$OPT -c $SRC -o $DEST/stencils_ppc64le_$OPT.o

# S390x
# s390x-linux-gnu-gcc-13 $FLAGS -$OPT -c $SRC -o $DEST/stencils_s390x_$OPT.o

# Mips (Big Endian)
mips-linux-gnu-gcc-13 $FLAGS -$OPT -c $SRC -o $DEST/stencils_mips_$OPT.o

# Mips (Little Endian)
mipsel-linux-gnu-gcc-13 $FLAGS -$OPT -c $SRC -o $DEST/stencils_mipsel_$OPT.o

# RISCV 32 Bit
# riscv64-linux-gnu-gcc-13 $FLAGS -$OPT -march=rv32imac -mabi=ilp32 -c $SRC -o $DEST/stencils_riscv_$OPT.o

# RISCV 64 Bit
riscv64-linux-gnu-gcc-13 $FLAGS -$OPT -c $SRC -o $DEST/stencils_riscv64_$OPT.o


# -------------- Cross compile runner --------------
mkdir -p build/runner

# Aarch64
aarch64-linux-gnu-gcc-13 -static -O3 -DENABLE_LOGGING -o build/runner/coparun-aarch64 src/coparun/runmem.c src/coparun/coparun.c src/coparun/mem_man.c



