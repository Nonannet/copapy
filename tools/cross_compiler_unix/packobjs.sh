#!/bin/sh

set -e
set -v

CC_NAME=$1
LD_NAME=$2
DEST_FILE=$3

echo "- Config musl"
./configure CFLAGS="-O2 -fno-stack-protector" CC=$CC_NAME

echo "- Build musl"
make clean
make all

mkdir -p ../build/stencil_objs

echo "- Extracting required objects"
cd ../build/stencil_objs
ar x ../../musl/lib/libc.a sinf.lo cosf.lo tanf.lo asinf.lo acosf.lo atanf.lo atan2f.lo
ar x ../../musl/lib/libc.a sqrtf.lo logf.lo expf.lo sqrt.lo
ar x ../../musl/lib/libc.a logf_data.lo __tandf.lo __cosdf.lo __sindf.lo
ar x ../../musl/lib/libc.a fabsf.lo scalbn.lo floor.lo exp2f_data.lo powf.lo powf_data.lo
ar x ../../musl/lib/libc.a __rem_pio2f.lo __math_invalidf.lo __stack_chk_fail.lo __math_divzerof.lo __math_oflowf.lo __rem_pio2_large.lo __math_uflowf.lo __math_xflowf.lo

cd ../../musl

echo "- Merge objects"
$LD_NAME -r ../build/stencil_objs/*.lo -o $DEST_FILE 

rm ../build/stencil_objs/*
