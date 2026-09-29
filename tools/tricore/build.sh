#!/bin/bash
# Builds the TriCore stencil object file and/or the bare metal runner for
# running tests in qemu-system-tricore (tools/tricore/run_qemu.sh).
#
# Usage: tools/tricore/build.sh [stencils|runner|all]
#
# The stencils are compiled from build/stencils/stencils.c
# (python stencils/generate_stencils.py build/stencils/stencils.c).
#
# Toolchain: https://github.com/NoMore201/tricore-gcc-toolchain/releases/download/11.3.1-20250101/tricore-gcc-11.3.1-20250101-linux.zip
# in PATH or in /opt/tricore

set -eu

TARGET=${1:-all}
PATH="$PATH:/opt/tricore/bin"

# AURIX TC2xx (TriCore 1.6.1), matches the qemu cpu tc27x
CPU_FLAGS="-mcpu=tc27xx"

if [[ "$TARGET" == "stencils" || "$TARGET" == "all" ]]; then
    echo "--------------tricore stencils----------------"
    mkdir -p build/stencils src/copapy/obj

    # -foptimize-sibling-calls: Not enabled by -O3 for TriCore, required for
    # jumps instead of calls to the result_* functions
    # --no-reloc: The assembler resolves branches to local labels
    # itself instead of emitting relocations for linker relaxation
    tricore-elf-gcc $CPU_FLAGS -fno-pic -ffunction-sections -foptimize-sibling-calls -Wa,--no-reloc \
        -c build/stencils/stencils.c -O3 -o build/stencils/stencils_tricore.o

    # Math functions from newlib. --unique=.rodata keeps the .rodata sections of the
    # libm objects separated, since ld calculates wrong addends of section
    # relative relocations when merging them
    tricore-elf-ld --mcpu=tc161 -r --unique=.rodata \
        build/stencils/stencils_tricore.o \
        $(tricore-elf-gcc $CPU_FLAGS -print-file-name=libm.a) \
        $(tricore-elf-gcc $CPU_FLAGS -print-libgcc-file-name) \
        -o src/copapy/obj/stencils_tricore_O3.o

    tricore-elf-objdump -d -x \
        src/copapy/obj/stencils_tricore_O3.o \
        > build/stencils/stencils_tricore_O3.asm
fi

if [[ "$TARGET" == "runner" || "$TARGET" == "all" ]]; then
    echo "--------------tricore runner----------------"
    mkdir -p build/runner

    tricore-elf-gcc $CPU_FLAGS -O2 -g \
        -Wall -Wextra -Wshadow \
        -DCOPARUN_BARE_METAL -DENABLE_BASIC_LOGGING \
        -T tools/tricore/memory-qemu.x \
        src/coparun/runmem.c \
        src/coparun/coparun_baremetal.c \
        src/coparun/mem_man.c \
        -o build/runner/coparun-tricore.elf
fi
