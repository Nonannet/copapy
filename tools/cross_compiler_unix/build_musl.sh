#!/bin/sh

set -e
set -v

mkdir -p /object_files

#git clone --single-branch --branch master --depth 1 https://git.musl-libc.org/git/musl
git clone --single-branch --branch master --depth 1 https://repo.or.cz/musl.git
cd musl

#./configure CFLAGS="-O2 -fno-stack-protector -ffast-math"

# x86_64
sh ../packobjs.sh gcc ld /object_files/musl_objects_x86_64.o

# x86
sh ../packobjs.sh i686-linux-gnu-gcc-13 i686-linux-gnu-ld /object_files/musl_objects_x86.o -fno-pic

# Arm64
sh ../packobjs.sh aarch64-linux-gnu-gcc-13 aarch64-linux-gnu-ld /object_files/musl_objects_arm64.o

# Armv6
sh ../packobjs.sh arm-none-eabi-gcc arm-none-eabi-ld /object_files/musl_objects_armv6.o "-march=armv6 -mfpu=vfp -mfloat-abi=hard -marm"

# Armv7
sh ../packobjs.sh arm-none-eabi-gcc arm-none-eabi-ld /object_files/musl_objects_armv7.o "-march=armv7-a -mfpu=neon-vfpv3 -mfloat-abi=hard -marm"

# Armv7 Thumb for Cortex-M3..7
sh ../packobjs.sh arm-none-eabi-gcc arm-none-eabi-ld /object_files/musl_objects_armv7thumb.o "-march=armv7-a -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb"

#sh ../packobjs.sh mips mips-linux-gnu-gcc-13 mips-linux-gnu-ld

#sh ../packobjs.sh riscv64 riscv64-linux-gnu-gcc-13 riscv64-linux-gnu-ld

echo "- clean up..."
rm -r ./*
cd ..
