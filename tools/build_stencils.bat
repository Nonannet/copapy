python stencils/generate_stencils.py build/stencils/stencils.c

echo -------------x86_64 - 64 bit-----------------

echo - Build stencils for 64 bit...
wsl mkdir -p build/stencil_objs
wsl rm build/stencil_objs/*

wsl gcc -fno-pic -c build/stencils/stencils.c -O3 -o build/stencil_objs/stencils.lo

cd build\stencil_objs
wsl ar x ../../../musl/lib/libc.a sinf.lo cosf.lo tanf.lo asinf.lo acosf.lo atanf.lo atan2f.lo
wsl ar x ../../../musl/lib/libc.a sqrtf.lo logf.lo expf.lo sqrt.lo
wsl ar x ../../../musl/lib/libc.a logf_data.lo __tandf.lo __cosdf.lo __sindf.lo __rem_pio2f.lo __math_invalidf.lo __stack_chk_fail.lo __math_divzerof.lo __math_oflowf.lo __rem_pio2_large.lo scalbn.lo floor.lo exp2f_data.lo powf.lo powf_data.lo __math_uflowf.lo __math_xflowf.lo
wsl ar x ../../../musl/lib/libc.a fabsf.lo

REM wsl ar t ../../../musl/lib/libc.a __math_xflowf.lo

wsl ld -r *.lo -o ../../src/copapy/obj/stencils_x86_64_O3.o

cd ..
cd ..
wsl objdump -d -x src/copapy/obj/stencils_x86_64_O3.o > build/stencils/stencils_x86_64_O3.asm


