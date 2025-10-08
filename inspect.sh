#!/bin/bash

source build.sh
python tests/test_compile_div.py
python tools/extract_code.py
objdump -D -b binary -m i386:x86-64 --adjust-vma=0x1000 bin/test_code.bin > bin/test_code.bin.txt
