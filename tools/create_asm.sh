#!/bin/bash

set -e
set -v

mkdir -p build/runner

cparch=$(python3 -c "import copapy; print(copapy._stencils.detect_process_arch())")

# Disassemble stencil object file
objdump -d -x src/copapy/obj/stencils_${cparch}_O3.o > build/runner/stencils.asm

# Create example code disassembly
python3 tools/make_example.py
build/runner/coparun build/runner/test.copapy build/runner/test.copapy.bin

if [ "$cparch" = 'x86_64' ]; then
	cparch="i386:x86-64"
elif [ "$cparch" = 'x86' ]; then
	cparch="i386"
elif [ "$cparch" = 'arm64' ]; then
	cparch="aarch64"
elif [ "$cparch" = 'armv6' ]; then
	cparch="arm"
elif [ "$cparch" = 'armv7' ]; then
	cparch="arm"
fi

echo "Archtitecture: '$cparch'"

objdump -D -b binary -m $cparch --adjust-vma=0x10000 build/runner/test.copapy.bin > build/runner/example.asm

rm build/runner/test.copapy.bin
