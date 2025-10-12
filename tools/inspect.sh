#!/bin/bash

source tools/build.sh
objdump -d -x src/copapy/obj/stencils_x86_64_O3.o > bin/stencils_x86_64_O3.asm
python tools/make_example.py
python tools/extract_code.py "bin/test.copapy" "bin/test.copapy.bin"
objdump -D -b binary -m i386:x86-64 --adjust-vma=0x1000 bin/test.copapy.bin > bin/test.copapy.asm
