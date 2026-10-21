#!/bin/bash
# Disassemble all local stencil object files for the docs build. Creates
# build/tmp/runner-linux-<arch>/stencils.asm, the same location the CI
# docs job downloads the runner artifacts to.

set -e

for obj in src/copapy/obj/stencils_*_O3.o; do
    arch=$(basename "$obj" _O3.o)
    arch=${arch#stencils_}

    case "$arch" in
        x86_64)  objdump=objdump ;;
        x86)     objdump=i686-linux-gnu-objdump ;;
        arm64)   objdump=aarch64-linux-gnu-objdump ;;
        arm*)    objdump=arm-none-eabi-objdump ;;
        riscv*)  objdump=riscv64-unknown-elf-objdump ;;
        tricore) objdump=/opt/tricore/bin/tricore-elf-objdump ;;
        *)       echo "Skipping $obj: unknown architecture '$arch'"; continue ;;
    esac

    if ! command -v "$objdump" > /dev/null; then
        echo "Skipping $arch: $objdump not found"
        continue
    fi

    mkdir -p "build/tmp/runner-linux-$arch"
    "$objdump" -d -x "$obj" > "build/tmp/runner-linux-$arch/stencils.asm"
    echo "Disassembled $obj -> build/tmp/runner-linux-$arch/stencils.asm"
done
