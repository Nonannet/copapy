#!/bin/bash

# Build arm-v7 runner and stencils
bash tools/build.sh arm-v7

# Build arm-v7-thumb stencils
bash tools/build.sh arm-v7-thumb

# Build arm-v7-thumb example code
export CP_TARGET_ARCH=armv7thumb
python3 tools/make_example.py
build/runner/coparun-armv7 build/runner/test.copapy build/runner/test.copapy.bin

arm-none-eabi-objdump -D -b binary -marm -M force-thumb --adjust-vma=0x1000000 build/runner/test.copapy.bin > build/runner/test.copapy-example-armv7thumb.asm

# Build arm-v7-thumb example code
export CP_TARGET_ARCH=armv7
python3 tools/make_example.py
build/runner/coparun-armv7 build/runner/test.copapy build/runner/test.copapy.bin

arm-none-eabi-objdump -D -b binary -marm --adjust-vma=0x1000000 build/runner/test.copapy.bin > build/runner/test.copapy-example-armv7.asm