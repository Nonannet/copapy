#!/bin/bash

# Runs the test suite with the native runner for each given stencil
# architecture (e.g. armv7 armv7thumb armv7mthumb). The runner and the
# assembly listings are stored in build/artifacts/runner-linux-<arch>.
# All architectures are tested, also if one of them fails.

set -v

STATUS=0

for ARCH in "$@"; do
    echo "-------------- $ARCH --------------"
    export CP_TARGET_ARCH=$ARCH

    pytest || STATUS=1
    bash tools/create_asm.sh || STATUS=1

    mkdir -p build/artifacts/runner-linux-$ARCH
    cp build/runner/* build/artifacts/runner-linux-$ARCH/
done

exit $STATUS
