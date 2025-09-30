#!/bin/bash
set -e
set -v
python src/copapy/generate_stencils.py
mkdir -p src/copapy/obj
gcc -c src/copapy/stencils.c -o src/copapy/obj/stencils_x86_64.o
gcc -c src/copapy/stencils.c -O1 -o src/copapy/obj/stencils_x86_64_O1.o
gcc -c src/copapy/stencils.c -O2 -o src/copapy/obj/stencils_x86_64_O2.o
gcc -c src/copapy/stencils.c -O3 -o src/copapy/obj/stencils_x86_64_O3.o

gcc -Wall -Wextra -Wconversion -Wsign-conversion -Wshadow -Wstrict-overflow -Werror -g src/runner/runmem2.c -o bin/runmem2
#x86_64-w64-mingw32-gcc -Wall -Wextra -Wconversion -Wsign-conversion -Wshadow -Wstrict-overflow -Werror src/runner/runmem2.c -Wall -O3 -o bin/runmem2.exe