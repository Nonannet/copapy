python tools/make_example.py
REM wsl qemu-aarch64 bin/coparun-aarch64 bin/test-aarch64.copapy bin/test-aarch64.copapy.bin
REM wsl aarch64-linux-gnu-objdump -D -b binary -m aarch64 --adjust-vma=0x5000 bin/test-aarch64.copapy.bin

REM wsl aarch64-linux-gnu-objdump -D -b binary -m aarch64 --adjust-vma=0x5000 bin/test-aarch64.copapy2.bin

bin\coparun bin/test.copapy bin/test.copapy.bin
wsl objdump -D -b binary -m i386:x86-64 --adjust-vma=0x500000 bin/test.copapy.bin

REM wsl aarch64-linux-gnu-objdump -d -x src/copapy/obj/stencils_arm64_O3.o

REM wsl objdump -D -b binary -m i386 --adjust-vma=0x5000 bin/test-x86.copapy.bin



REM wsl objdump -d -x src/copapy/obj/stencils_x86_O3.o