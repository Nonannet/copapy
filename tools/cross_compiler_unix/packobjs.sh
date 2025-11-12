#!/bin/sh

ARCH=$1

mkdir -p ../build/stencil_objs

cd ../build/stencil_objs
ar x ../../../musl/lib/libc.a sinf.lo cosf.lo tanf.lo asinf.lo acosf.lo atanf.lo atan2f.lo
ar x ../../../musl/lib/libc.a sqrtf.lo logf.lo expf.lo sqrt.lo
ar x ../../../musl/lib/libc.a logf_data.lo __tandf.lo __cosdf.lo __sindf.lo __rem_pio2f.lo __math_invalidf.lo __stack_chk_fail.lo __math_divzerof.lo __math_oflowf.lo __rem_pio2_large.lo scalbn.lo floor.lo exp2f_data.lo powf.lo powf_data.lo __math_uflowf.lo __math_xflowf.lo
ar x ../../../musl/lib/libc.a fabsf.lo

ld -r *.lo -o ../musl_objects_{$ARCH}.o

rm ../build/stencil_objs/*
cd ../../musl