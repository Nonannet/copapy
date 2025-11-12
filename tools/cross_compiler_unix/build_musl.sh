
#!/bin/sh

git clone --single-branch --branch master --depth 1 https://git.musl-libc.org/git/musl
cd musl

./configure CFLAGS="-O2 -fno-stack-protector -ffast-math"
sh packobjs.sh x86_64

./configure CFLAGS="-O2 -fno-stack-protector" CC=i686-linux-gnu-gcc-13
sh packobjs.sh x86

./configure CFLAGS="-O2 -fno-stack-protector" CC=aarch64-linux-gnu-gcc-13
sh packobjs.sh arm64

#./configure CFLAGS="-O2 -fno-stack-protector" CC=mips-linux-gnu-gcc-13
#sh packobjs.sh mips

#./configure CFLAGS="-O2 -fno-stack-protector" CC=riscv64-linux-gnu-gcc-13
#sh packobjs.sh riscv64



