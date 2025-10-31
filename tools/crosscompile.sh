#!/bin/bash

set -e
set -v

mkdir -p bin
SRC=bin/stencils.c
DEST=src/copapy/obj
OPT=O3
FLAGS="-fno-pic -fno-plt"

mkdir -p $DEST

# -------------- Compile stencils --------------

# Windows x86_64 (ARM64)
python3 stencils/generate_stencils.py --abi ms $SRC
gcc-13 $FLAGS -$OPT -c $SRC -o $DEST/stencils_AMD64_$OPT.o

# Windows x86
gcc-13 $FLAGS -m32 -$OPT -c $SRC -o $DEST/stencils_x86_$OPT.o


# Native x86_64
python3 stencils/generate_stencils.py $SRC
gcc-13 $FLAGS -$OPT -c $SRC -o $DEST/stencils_x86_64_$OPT.o

# Native i686
gcc-13 $FLAGS -m32 -$OPT -c $SRC -o $DEST/stencils_i686_$OPT.o

# ARM64 linux (aarch64)
aarch64-linux-gnu-gcc-13 $FLAGS -$OPT -c $SRC -o $DEST/stencils_aarch64_$OPT.o

# ARM64 macos (copy aarch64)
cp $DEST/stencils_aarch64_$OPT.o $DEST/stencils_arm64_$OPT.o

# ARMv7
arm-linux-gnueabihf-gcc-13 $FLAGS -$OPT -c $SRC -o $DEST/stencils_armv7_$OPT.o

# PowerPC64LE
# powerpc64le-linux-gnu-gcc-13 $FLAGS -$OPT -c $SRC -o $DEST/stencils_ppc64le_$OPT.o

# S390x
# s390x-linux-gnu-gcc-13 $FLAGS -$OPT -c $SRC -o $DEST/stencils_s390x_$OPT.o

# Mips (Big Endian)
mips-linux-gnu-gcc-13 $FLAGS -$OPT -c $SRC -o $DEST/stencils_mips_$OPT.o

# Mips (Little Endian)
mipsel-linux-gnu-gcc-13 $FLAGS -$OPT -c $SRC -o $DEST/stencils_mipsel_$OPT.o

# RISCV 32 Bit
# riscv64-linux-gnu-gcc-13 $FLAGS -$OPT -march=rv32imac -mabi=ilp32 -c $SRC -o $DEST/stencils_riscv32_$OPT.o

# RISCV 64 Bit
riscv64-linux-gnu-gcc-13 $FLAGS -$OPT -c $SRC -o $DEST/stencils_riscv64_$OPT.o


# -------------- Cross compile runner --------------

# Aarch64
aarch64-linux-gnu-gcc-13 -static -O3 -o bin/coparun-aarch64 src/coparun/runmem.c src/coparun/coparun.c src/coparun/mem_man.c
