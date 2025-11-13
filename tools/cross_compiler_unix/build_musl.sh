
#!/bin/sh

set -e
set -v

git clone --single-branch --branch master --depth 1 https://git.musl-libc.org/git/musl
cd musl

#./configure CFLAGS="-O2 -fno-stack-protector -ffast-math"

sh ../packobjs.sh x86_64 gcc ld

sh ../packobjs.sh x86 i686-linux-gnu-gcc-13 i686-linux-gnu-ld

sh ../packobjs.sh arm64 aarch64-linux-gnu-gcc-13 aarch64-linux-gnu-ld

#sh ../packobjs.sh mips mips-linux-gnu-gcc-13 mips-linux-gnu-ld

#sh ../packobjs.sh riscv64 riscv64-linux-gnu-gcc-13 riscv64-linux-gnu-ld

echo "- clean up..."
rm -r ./*
cd ..
