#!/bin/bash

source tools/build.sh

objdump -d -j .text src/copapy/obj/stencils_x86_64_O3.o > build/stencils/stencils_x86_64_O3.asm

python3 tools/make_example.py
python3 tools/extract_code.py "build/runner/test.copapy" "build/runner/test.copapy.bin"
objdump -D -b binary -m i386:x86-64 --adjust-vma=0x1000 build/runner/test.copapy.bin > build/runner/test.copapy.asm
