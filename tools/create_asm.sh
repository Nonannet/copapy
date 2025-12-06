#!/bin/bash

set -e
set -v

mkdir -p build/runner

arch=$(python3 -c "import copapy; print(copapy._stencils.detect_process_arch())")

# Disassemble stencil object file
objdump -d -x src/copapy/obj/stencils_${arch}_O3.o > build/runner/stencils.asm

# Create example code disassembly
python3 tools/make_example.py
build/runner/coparun build/runner/test.copapy build/runner/test.copapy.bin

if [ $(arch) = 'x86-64' ]; then
	arch="i386:x86-64"
fi

objdump -D -b binary -m $arch --adjust-vma=0x10000 build/runner/test.copapy.bin > build/runner/example.asm

rm build/runner/test.copapy.bin
