#!/bin/sh

set -e
set -v

# Sources of the CMSIS-DSP math functions for the Cortex-M stencils, compiled
# by tools/build_cmsis_libm.sh together with stencils/cmsis_libm.c
CMSIS_DSP=/opt/CMSIS-DSP

mkdir -p /object_files

git clone --single-branch --branch v1.18.0 --depth 1 --filter=blob:none --sparse https://github.com/ARM-software/CMSIS-DSP.git $CMSIS_DSP
cd $CMSIS_DSP
git sparse-checkout set Include PrivateInclude Source/FastMathFunctions Source/CommonTables ComputeLibrary

cp ./LICENSE /object_files/CMSIS-DSP-LICENSE
cp ./ComputeLibrary/LICENSE.txt /object_files/CMSIS-DSP-ComputeLibrary-LICENSE
