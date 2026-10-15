#!/bin/bash
# Builds the Zephyr runner (targets/zephyr) for a board and copies the
# ELF file to build/runner/coparun-zephyr-<board>.elf
#
# Requires a Zephyr environment: ZEPHYR_BASE set, cmake and ninja available
# and a toolchain (Zephyr SDK or ZEPHYR_TOOLCHAIN_VARIANT).
#
# Usage: tools/build_zephyr.sh [board] [additional cmake arguments]
#   tools/build_zephyr.sh mps2/an500                      (Cortex-M7 in qemu)
#   tools/build_zephyr.sh teensy41                        (runner on USB serial)
set -eu

BOARD=${1:-mps2/an500}
shift || true
NAME=${BOARD//\//_}
BUILD_DIR=build/zephyr/$NAME

cmake -GNinja -B $BUILD_DIR -S targets/zephyr -DBOARD=$BOARD "$@"
ninja -C $BUILD_DIR

mkdir -p build/runner
cp $BUILD_DIR/zephyr/zephyr.elf build/runner/coparun-zephyr-$NAME.elf
