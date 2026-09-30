#!/bin/bash

set -e
set -v

RUNNER="${RUNNER:-build/runner/coparun}"
OBJDUMP="${OBJDUMP:-objdump}"

mkdir -p build/runner

cparch=$(python3 -c "import copapy; print(copapy._stencils.detect_process_arch())")

# Disassemble stencil object file
$OBJDUMP -d -x src/copapy/obj/stencils_${cparch}_O3.o > build/runner/stencils.asm

python3 tools/make_example.py
$RUNNER build/runner/test.copapy build/runner/test.copapy.bin

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
elif [ "$cparch" = 'riscv64' ]; then
	cparch="riscv:rv64"
elif [ "$cparch" = 'riscv32' ]; then
	cparch="riscv:rv32"
fi

echo "Archtitecture: '$cparch'"

if [[ "$cparch" == *"thumb"* ]]; then
	$OBJDUMP -D -b binary -marm -M force-thumb --adjust-vma=0x10000 build/runner/test.copapy.bin > build/runner/example.asm
else
	$OBJDUMP -D -b binary -m $cparch --adjust-vma=0x10000 build/runner/test.copapy.bin > build/runner/example.asm
fi

rm build/runner/test.copapy.bin
