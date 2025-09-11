#!/bin/bash
set -e
set -v
python src/copapy/generate_stencils.py
mkdir -p src/copapy/obj
gcc -c src/copapy/stencils.c -o src/copapy/obj/stencils_x86_64.o
ls -l src/copapy/obj/*.o
