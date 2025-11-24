#!/bin/bash

set -e
set -v

sh tools/build.sh

#objdump -d -j .text src/copapy/obj/stencils_x86_64_O3.o > build/stencils/stencils_x86_64_O3.asm

python3 tools/make_example.py

build/runner/coparun build/runner/test.copapy build/runner/test.copapy.bin
objdump -D -b binary -m i386:x86-64 --adjust-vma=0x1000 build/runner/test.copapy.bin > build/runner/test.copapy.asm

build/runner/coparun-armv7 build/runner/test-armv7.copapy build/runner/test.copapy-armv7.bin
arm-none-eabi-objdump -D -b binary -marm --adjust-vma=0x50000 build/runner/test.copapy-armv7.bin > build/runner/test.copapy-armv7.asm
