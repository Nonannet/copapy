#!/bin/bash

source tools/build.sh

objdump --disassembler-color=on -d -j .text src/copapy/obj/stencils_x86_64_O3.o \
| ansi2html -i > bin/stencils_x86_64_O3.html

python3 tools/make_example.py
python3 tools/extract_code.py "bin/test.copapy" "bin/test.copapy.bin"
objdump --disassembler-color=on -D -b binary -m i386:x86-64 --adjust-vma=0x1000 bin/test.copapy.bin \
| bin/test.copapy.asm | ansi2html -i > bin/test.copapy.html