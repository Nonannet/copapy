#!/bin/bash
set -eu

ARCH=${1:-x86_64}

case "$ARCH" in
    (x86_64|x86|arm64|arm-v6|arm-v7|arm-v7-thumb|arm-v7m-thumb|all)
        ;;
    (*)
        echo "Usage: $0 [x86_64|x86|arm64|arm-v6|arm-v7|arm-v7-thumb|arm-v7m-thumb|all]"
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
# x86 32-bit
#######################################
if [[ "$ARCH" == "x86" || "$ARCH" == "all" ]]; then
    echo "--------------x86 32 bit----------------"

    if command -v i686-linux-gnu-gcc >/dev/null 2>&1; then
        X86_CC=i686-linux-gnu-gcc
        X86_LD=i686-linux-gnu-ld
        X86_OBJDUMP=i686-linux-gnu-objdump
        X86_EXTRA_FLAGS=""
        X86_LD_FLAGS=""
    elif gcc -m32 -x c -o /tmp/copapy_x86_probe - <<'EOF' >/dev/null 2>&1
int main(void) { return 0; }
EOF
    then
        X86_CC=gcc
        X86_LD=ld
        X86_OBJDUMP=objdump
        X86_EXTRA_FLAGS="-m32"
        X86_LD_FLAGS="-m elf_i386"
    else
        echo "x86 32-bit toolchain not available: install gcc-multilib or i686-linux-gnu-gcc" >&2
        exit 1
    fi

    "$X86_CC" $X86_EXTRA_FLAGS -fno-pic -ffunction-sections \
        -c $SRC -O3 -o build/stencils/stencils.o
    "$X86_LD" $X86_LD_FLAGS -r build/stencils/stencils.o \
        build/musl/musl_objects_x86.o \
        -o $DEST/stencils_x86_O3.o
    "$X86_OBJDUMP" -d -x $DEST/stencils_x86_O3.o \
        > build/stencils/stencils_x86_O3.asm

    "$X86_CC" $X86_EXTRA_FLAGS -static -O3 -DENABLE_LOGGING \
        src/coparun/runmem.c \
        src/coparun/coparun.c \
        src/coparun/mem_man.c \
        -o build/runner/coparun-x86
fi

#######################################
# ARM 64
#######################################
if [[ "$ARCH" == "arm64" || "$ARCH" == "all" ]]; then
    echo "--------------arm64----------------"

    LIBGCC=$(aarch64-linux-gnu-gcc -print-libgcc-file-name)

    aarch64-linux-gnu-gcc -fno-pic -ffunction-sections \
        -c $SRC -O3 -o build/stencils/stencils.o

    aarch64-linux-gnu-ld -r \
        build/stencils/stencils.o \
        build/musl/musl_objects_arm64.o \
        $LIBGCC \
        -o $DEST/stencils_arm64_O3.o

    aarch64-linux-gnu-objdump -d -x \
        $DEST/stencils_arm64_O3.o \
        > build/stencils/stencils_arm64_O3.asm

    aarch64-linux-gnu-gcc \
        -Wall -Wextra -Wconversion -Wsign-conversion -static \
        -Wshadow -Wstrict-overflow -O3 \
        -DENABLE_LOGGING \
        src/coparun/runmem.c \
        src/coparun/coparun.c \
        src/coparun/mem_man.c \
        -o build/runner/coparun-arm64
fi

#######################################
# ARM v6
#######################################
if [[ "$ARCH" == "arm-v6" || "$ARCH" == "all" ]]; then
    echo "--------------arm-v6 32 bit----------------"

    LIBGCC=$(arm-none-eabi-gcc -march=armv6 -mfpu=vfp -mfloat-abi=hard -marm -print-libgcc-file-name)

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

    LIBGCC=$(arm-none-eabi-gcc -march=armv7-a -mfpu=neon-vfpv3 -mfloat-abi=hard -marm -print-libgcc-file-name)

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

    # The same runner for all ARM7
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

#######################################
# ARM v7 thumb Cortex-A
#######################################
if [[ "$ARCH" == "arm-v7-thumb" || "$ARCH" == "all" ]]; then
    echo "--------------arm-v7a-thumb 32 bit----------------"

    LIBGCC=$(arm-none-eabi-gcc -march=armv7 -mfpu=vfp3 -mthumb -print-libgcc-file-name)

    arm-none-eabi-gcc -fno-pic -ffunction-sections \
        -march=armv7-a -mfpu=neon-vfpv3 -mfloat-abi=hard -mthumb \
        -c $SRC -O3 -o build/stencils/stencils.o

    arm-none-eabi-ld -r \
        build/stencils/stencils.o \
        build/musl/musl_objects_armv7thumb.o \
        $LIBGCC \
        -o $DEST/stencils_armv7thumb_O3.o

    arm-none-eabi-objdump -d -x \
        $DEST/stencils_armv7thumb_O3.o \
        > build/stencils/stencils_armv7thumb_O3.asm

    # The same runner for all ARM7
    arm-linux-gnueabihf-gcc \
        -march=armv7-a -mfpu=neon-vfpv3 -mfloat-abi=hard -static \
        -Wall -Wextra -Wconversion -Wsign-conversion \
        -Wshadow -Wstrict-overflow -O3 \
        -DENABLE_LOGGING \
        src/coparun/runmem.c \
        src/coparun/coparun.c \
        src/coparun/mem_man.c \
        -o build/runner/coparun-armv7thumb
fi

#######################################
# ARM v7 thumb Cortex-M
#######################################
if [[ "$ARCH" == "arm-v7m-thumb" || "$ARCH" == "all" ]]; then
    echo "--------------arm-v7m-thumb 32 bit----------------"

    LIBGCC=$(arm-none-eabi-gcc -march=armv7e-m -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -print-libgcc-file-name)

    arm-none-eabi-gcc -fno-pic -ffunction-sections \
        -march=armv7e-m -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb \
        -c $SRC -O3 -o build/stencils/stencils.o

    arm-none-eabi-ld -r \
        build/stencils/stencils.o \
        build/musl/musl_objects_armv7mthumb.o \
        $LIBGCC \
        -o $DEST/stencils_armv7mthumb_O3.o

    arm-none-eabi-objdump -d -x \
        $DEST/stencils_armv7mthumb_O3.o \
        > build/stencils/stencils_armv7mthumb_O3.asm

    # The same runner for all ARM7
    arm-linux-gnueabihf-gcc \
        -march=armv7-a -mfpu=neon-vfpv3 -mfloat-abi=hard -static \
        -Wall -Wextra -Wconversion -Wsign-conversion \
        -Wshadow -Wstrict-overflow -O3 \
        -DENABLE_LOGGING \
        src/coparun/runmem.c \
        src/coparun/coparun.c \
        src/coparun/mem_man.c \
        -o build/runner/coparun-armv7thumb
fi