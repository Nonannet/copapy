python tools/make_example.py
REM python tools/extract_code.py "bin/test.copapy" "bin/test.copapy.bin"
wsl qemu-aarch64 bin/coparun-aarch64 bin/test-aarch64.copapy bin/test-aarch64.copapy.bin
wsl aarch64-linux-gnu-objdump -D -b binary -m aarch64 --adjust-vma=0x5000 bin/test-aarch64.copapy.bin

python tools/extract_code.py "bin/test-aarch64.copapy" "bin/test-aarch64.copapy2.bin"
wsl aarch64-linux-gnu-objdump -D -b binary -m aarch64 --adjust-vma=0x5000 bin/test-aarch64.copapy2.bin

REM wsl objdump -D -b binary -m i386:x86-64 --adjust-vma=0x1000 bin/test.copapy.bin

REM wsl aarch64-linux-gnu-objdump -d -x src/copapy/obj/stencils_aarch64_O3.o
