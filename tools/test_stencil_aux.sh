#!/bin/bash

set -e
set -v

FILE=test
SRC="stencils/$FILE.c"
DEST=bin
OPT=O3
CMSIS_DSP=${CMSIS_DSP:-/opt/CMSIS-DSP}

mkdir -p $DEST

# Compile native x86_64, the math functions used by the aux functions are in libm.
# The math functions of the Cortex-M stencils based on CMSIS-DSP are tested as well.
gcc -g -$OPT -DTEST_CMSIS_LIBM -D__GNUC_PYTHON__ -I$CMSIS_DSP/Include -I$CMSIS_DSP/PrivateInclude \
    $SRC \
    $CMSIS_DSP/Source/FastMathFunctions/arm_sin_f32.c \
    $CMSIS_DSP/Source/FastMathFunctions/arm_cos_f32.c \
    $CMSIS_DSP/Source/FastMathFunctions/arm_atan2_f32.c \
    $CMSIS_DSP/Source/CommonTables/arm_common_tables.c \
    -o $DEST/$FILE -lm
chmod +x $DEST/$FILE

# Run
$DEST/$FILE
