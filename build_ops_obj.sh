#!/bin/bash
set -e
set -v
python src/copapy/generate_stancils.py
mkdir -p src/copapy/obj
gcc -c src/copapy/stancils.c -o src/copapy/obj/stancils_x86_64.o
ls -l src/copapy/obj/*.o
