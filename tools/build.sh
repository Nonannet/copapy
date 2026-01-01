#!/bin/bash
set -eux

ARCH=${1:-x86_64}

case "$ARCH" in
    (x86_64|arm-v6|arm-v7|all)
        ;;
    (*)
        echo "Usage: $0 [x86_64|arm-v6|arm-v7|all]"
        exit 1
        ;;
esac

mkdir -p build/stencils
mkdir -p build/runner

SRC=build/stencils/stencils.c
DEST=src/copapy/obj
python3 stencils/generate_stencils.py $SRC
mkdir -p $DEST

#######################################
# x86_64
#######################################
if [[ "$ARCH" == "x86_64" || "$ARCH" == "all" ]]; then
    echo "--------------x86_64----------------"

    gcc -fno-pic -ffunction-sections -c $SRC -O3 -o build/stencils/stencils.o
    ld -r build/stencils/stencils.o build/musl/musl_objects_x86_64.o \
        -o $DEST/stencils_x86_64_O3.o
    objdump -d -x $DEST/stencils_x86_64_O3.o \
        > build/stencils/stencils_x86_64_O3.asm

    mkdir -p bin
    gcc -Wall -Wextra -Wconversion -Wsign-conversion \
        -Wshadow -Wstrict-overflow -Werror -g -O3 \
        -DENABLE_LOGGING \
        src/coparun/runmem.c \
        src/coparun/coparun.c \
        src/coparun/mem_man.c \
        -o build/runner/coparun
fi

#######################################
# ARM v6
#######################################
if [[ "$ARCH" == "arm-v6" || "$ARCH" == "all" ]]; then
    echo "--------------arm-v6 32 bit----------------"

    LIBGCC=$(arm-none-eabi-gcc -print-libgcc-file-name)

    arm-none-eabi-gcc -fno-pic -ffunction-sections \
        -march=armv6 -mfpu=vfp -mfloat-abi=hard -marm \
        -c $SRC -O3 -o build/stencils/stencils.o

    arm-none-eabi-ld -r \
        build/stencils/stencils.o \
        build/musl/musl_objects_armv6.o \
        $LIBGCC \
        -o $DEST/stencils_armv6_O3.o

    arm-none-eabi-objdump -d -x \
        $DEST/stencils_armv6_O3.o \
        > build/stencils/stencils_armv6_O3.asm

    arm-linux-gnueabihf-gcc \
        -march=armv6 -mfpu=vfp -mfloat-abi=hard -marm -static \
        -Wall -Wextra -Wconversion -Wsign-conversion \
        -Wshadow -Wstrict-overflow -O3 \
        -DENABLE_LOGGING \
        src/coparun/runmem.c \
        src/coparun/coparun.c \
        src/coparun/mem_man.c \
        -o build/runner/coparun-armv6
fi

#######################################
# ARM v7
#######################################
if [[ "$ARCH" == "arm-v7" || "$ARCH" == "all" ]]; then
    echo "--------------arm-v7 32 bit----------------"

    LIBGCC=$(arm-none-eabi-gcc -print-libgcc-file-name)

    arm-none-eabi-gcc -fno-pic -ffunction-sections \
        -march=armv7-a -mfpu=neon-vfpv3 -mfloat-abi=hard -marm \
        -c $SRC -O3 -o build/stencils/stencils.o

    arm-none-eabi-ld -r \
        build/stencils/stencils.o \
        build/musl/musl_objects_armv7.o \
        $LIBGCC \
        -o $DEST/stencils_armv7_O3.o

    arm-none-eabi-objdump -d -x \
        $DEST/stencils_armv7_O3.o \
        > build/stencils/stencils_armv7_O3.asm

    arm-linux-gnueabihf-gcc \
        -march=armv7-a -mfpu=neon-vfpv3 -mfloat-abi=hard -marm -static \
        -Wall -Wextra -Wconversion -Wsign-conversion \
        -Wshadow -Wstrict-overflow -O3 \
        -DENABLE_LOGGING \
        src/coparun/runmem.c \
        src/coparun/coparun.c \
        src/coparun/mem_man.c \
        -o build/runner/coparun-armv7
fi
