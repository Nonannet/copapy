#!/bin/bash

set -e
set -v

FILE=test
SRC="stencils/$FILE.c"
DEST=bin
OPT=O3

mkdir -p $DEST

# Compile native x86_64, the math functions used by the aux functions are in libm
gcc -g -$OPT $SRC -o $DEST/$FILE -lm
chmod +x $DEST/$FILE

# Run
$DEST/$FILE

# Math functions of the Cortex-M stencils based on CMSIS-DSP, tested on x86_64
CMSIS_DSP=${CMSIS_DSP:-/opt/CMSIS-DSP}
FILE=test_cmsis_libm
gcc -g -$OPT -D__GNUC_PYTHON__ -I$CMSIS_DSP/Include -I$CMSIS_DSP/PrivateInclude \
    stencils/$FILE.c \
    $CMSIS_DSP/Source/FastMathFunctions/arm_sin_f32.c \
    $CMSIS_DSP/Source/FastMathFunctions/arm_cos_f32.c \
    $CMSIS_DSP/Source/FastMathFunctions/arm_atan2_f32.c \
    $CMSIS_DSP/Source/CommonTables/arm_common_tables.c \
    -lm -o $DEST/$FILE
chmod +x $DEST/$FILE

$DEST/$FILE