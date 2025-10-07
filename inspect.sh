#!/bin/bash

objdump -D -b binary -m i386:x86-64 --adjust-vma=0x1000 bin/test_code.bin