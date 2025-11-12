python tools/make_example.py
REM wsl qemu-aarch64 build/runner/coparun-aarch64 build/runner/test-arm64.copapy build/runner/test-arm64.copapy.bin
REM wsl aarch64-linux-gnu-objdump -D -b binary -m aarch64 --adjust-vma=0x5000 build/runner/test-arm64.copapy.bin

REM wsl aarch64-linux-gnu-objdump -D -b binary -m aarch64 --adjust-vma=0x5000 build/runner/test-arm64.copapy2.bin

bin\coparun build/runner/test.copapy build/runner/test.copapy.bin
wsl objdump -D -b binary -m i386:x86-64 --adjust-vma=0x500000 build/runner/test.copapy.bin

REM wsl aarch64-linux-gnu-objdump -d -x src/copapy/obj/stencils_arm64_O3.o

REM wsl objdump -D -b binary -m i386 --adjust-vma=0x5000 build/runner/test-x86.copapy.bin



REM wsl objdump -d -x src/copapy/obj/stencils_x86_O3.o