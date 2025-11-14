#!/bin/sh

set -e
set -v

CC_NAME=$1
LD_NAME=$2
DEST_FILE=$3
OPT_FLAGS=$4

echo "- Config musl"
./configure CFLAGS="-O2 -fno-stack-protector $OPT_FLAGS" CC=$CC_NAME

echo "- Build musl"
make clean
make all

mkdir -p ../build/stencil_objs

echo "- Extracting required objects"
cd ../build/stencil_objs

# Check out .o (non PIC)
ar x ../../musl/lib/libc.a sinf.o cosf.o tanf.o asinf.o acosf.o atanf.o atan2f.o
ar x ../../musl/lib/libc.a sqrtf.o logf.o expf.o sqrt.o
ar x ../../musl/lib/libc.a logf_data.o __tandf.o __cosdf.o __sindf.o
ar x ../../musl/lib/libc.a fabsf.o scalbn.o floor.o exp2f_data.o powf.o powf_data.o
ar x ../../musl/lib/libc.a __rem_pio2f.o __math_invalidf.o __stack_chk_fail.o __math_divzerof.o __math_oflowf.o __rem_pio2_large.o __math_uflowf.o __math_xflowf.o

# Check out .lo (PIC)
ar x ../../musl/lib/libc.a sinf.lo cosf.lo tanf.lo asinf.lo acosf.lo atanf.lo atan2f.lo
ar x ../../musl/lib/libc.a sqrtf.lo logf.lo expf.lo sqrt.lo
ar x ../../musl/lib/libc.a logf_data.lo __tandf.lo __cosdf.lo __sindf.lo
ar x ../../musl/lib/libc.a fabsf.lo scalbn.lo floor.lo exp2f_data.lo powf.lo powf_data.lo
ar x ../../musl/lib/libc.a __rem_pio2f.lo __math_invalidf.lo __stack_chk_fail.lo __math_divzerof.lo __math_oflowf.lo __rem_pio2_large.lo __math_uflowf.lo __math_xflowf.lo

cd ../../musl

echo "- Merge objects"
$LD_NAME -r ../build/stencil_objs/* -o $DEST_FILE 

rm ../build/stencil_objs/*
