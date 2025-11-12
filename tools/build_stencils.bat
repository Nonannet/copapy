python stencils/generate_stencils.py bin/stencils.c

echo -------------x86_64 - 64 bit-----------------

echo - Build stencils for 64 bit...
wsl mkdir -p build/stencil_objs
wsl gcc -fno-pic -c bin/stencils.c -O3 -o build/stencil_objs/stencils.lo

cd build\stencil_objs
wsl ar x /usr/lib/x86_64-linux-musl/libc.a sinf.lo cosf.lo tanf.lo sqrtf.lo logf.lo expf.lo logf_data.lo __tandf.lo __cosdf.lo __sindf.lo __rem_pio2f.lo __math_invalidf.lo __stack_chk_fail.lo __math_divzerof.lo __math_oflowf.lo __rem_pio2_large.lo scalbn.lo floor.lo
wsl ld -r *.lo -o ../../src/copapy/obj/stencils_x86_64_O3.o

cd ..
cd ..
wsl objdump -d -x src/copapy/obj/stencils_x86_64_O3.o > bin/stencils_x86_64_O3.asm