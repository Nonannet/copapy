#!/bin/bash

bash tools/build.sh arm-v7-thumb
python tests/test_ops_armv7thumb.py
qemu-arm -d in_asm -D qemu.log build/runner/coparun-armv7thumb build/runner/test-armv7thumb.copapy build/runner/test.copapy-armv7thumb.bin
arm-none-eabi-objdump -D -b binary -marm -M force-thumb --adjust-vma=0xff7ed000 build/runner/test.copapy-armv7thumb.bin > build/runner/test.copapy-armv7thumb.asm