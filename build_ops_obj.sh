#!/bin/bash
set -e

mkdir -p src/copapy/obj
gcc -c src/copapy/ops.c -o src/copapy/obj/ops_x86_64.o