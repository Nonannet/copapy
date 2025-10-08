#!/bin/bash

source tools/build.sh
python tests/test_compile_div.py
python tools/extract_code.py "bin/test.copapy" "bin/test.copapy.bin"
objdump -D -b binary -m i386:x86-64 --adjust-vma=0x1000 bin/test.copapy.bin > bin/test.copapy.asm
