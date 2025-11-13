#!/bin/sh

set -e
set -v

CC_NAME=$1
LD_NAME=$2
DEST_FILE=$3

echo "- Config musl"
./configure CFLAGS="-O2 -fno-pic -fno-stack-protector" CC=$CC_NAME

echo "- Build musl"
make clean
make all

mkdir -p ../build/stencil_objs

echo "- Extracting required objects"
cd ../build/stencil_objs
ar x ../../musl/lib/libc.a sinf.o cosf.o tanf.o asinf.o acosf.o atanf.o atan2f.o
ar x ../../musl/lib/libc.a sqrtf.o logf.o expf.o sqrt.o
ar x ../../musl/lib/libc.a logf_data.o __tandf.o __cosdf.o __sindf.o
ar x ../../musl/lib/libc.a fabsf.o scalbn.o floor.o exp2f_data.o powf.o powf_data.o
ar x ../../musl/lib/libc.a __rem_pio2f.o __math_invalidf.o __stack_chk_fail.o __math_divzerof.o __math_oflowf.o __rem_pio2_large.o __math_uflowf.o __math_xflowf.o

cd ../../musl

echo "- Merge objects"
$LD_NAME -r ../build/stencil_objs/*.o -o $DEST_FILE 

rm ../build/stencil_objs/*
