#!/bin/bash

# Builds the math functions for Cortex-M stencils: the CMSIS-DSP based functions
# of stencils/cmsis_libm.c and the remaining functions (sqrtf, floorf, ...)
# from the precompiled MUSL objects.
#
# Usage: build_cmsis_libm.sh <CMSIS-DSP path> <MUSL object file> <output file> <compiler flags>

set -e

CMSIS_DSP=$1
MUSL_OBJ=$2
DEST_FILE=$3
ARCH_FLAGS=$4

FUNCTIONS="sinf cosf tanf atanf atan2f asinf acosf expf logf powf"

SOURCES="stencils/cmsis_libm.c
    $CMSIS_DSP/Source/FastMathFunctions/arm_sin_f32.c
    $CMSIS_DSP/Source/FastMathFunctions/arm_cos_f32.c
    $CMSIS_DSP/Source/FastMathFunctions/arm_atan2_f32.c
    $CMSIS_DSP/Source/CommonTables/arm_common_tables.c"

TMP=$(mktemp -d)

# __GNUC_PYTHON__ removes the dependency on the CMSIS-Core headers
for SRC in $SOURCES; do
    arm-none-eabi-gcc $ARCH_FLAGS -O3 -fno-pic -ffunction-sections -fdata-sections \
        -D__GNUC_PYTHON__ -I$CMSIS_DSP/Include -I$CMSIS_DSP/PrivateInclude \
        -c $SRC -o $TMP/$(basename $SRC .c).o
done

# Keep only the sections used by the math functions (arm_common_tables.c
# contains the tables of all CMSIS-DSP functions)
arm-none-eabi-ld -r --gc-sections $(printf -- '-u %s ' $FUNCTIONS) $TMP/*.o -o $TMP/cmsis_libm.obj

# The CMSIS-DSP based functions replace the MUSL functions of the same name
arm-none-eabi-objcopy $(printf -- '--weaken-symbol=%s ' $FUNCTIONS) $MUSL_OBJ $TMP/musl.obj

arm-none-eabi-ld -r $TMP/cmsis_libm.obj $TMP/musl.obj -o $DEST_FILE

rm -r $TMP
