echo ------------------------------
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
echo - Build runner for Windows 64 bit...
cl /Zi /Od /DENABLE_BASIC_LOGGING src\coparun\runmem.c src\coparun\coparun.c src\coparun\mem_man.c /Fe:bin\coparun.exe

REM Optimized:
REM cl /O2 src\coparun\runmem.c src\coparun\coparun.c src\coparun\mem_man.c /Fe:bin\coparun.exe

echo - Build stencils for Windows 64 bit...
python stencils/generate_stencils.py --abi ms bin/stencils.c
wsl gcc -fno-pic -c bin/stencils.c -O3 -o src/copapy/obj/stencils_AMD64_O3.o

echo ------------------------------
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x86

echo - Build runner for Windows 32 bit...
cl /Zi /Od /DENABLE_BASIC_LOGGING src\coparun\runmem.c src\coparun\coparun.c src\coparun\mem_man.c /Fe:bin\coparun-x86.exe

echo - Build stencils for Windows 32 bit...
wsl gcc -m32 -fno-pic -c bin/stencils.c -O3 -o src/copapy/obj/stencils_x86_O3.o


echo ------------------------------
echo - Build stencils for aarch64...
python stencils/generate_stencils.py bin/stencils.c
wsl aarch64-linux-gnu-gcc -fno-pic -c bin/stencils.c -O3 -o src/copapy/obj/stencils_aarch64_O3.o

echo ------------------------------
echo - Build runner for Aarch64...
wsl aarch64-linux-gnu-gcc -static -Wall -Wextra -Wconversion -Wsign-conversion -Wshadow -Wstrict-overflow -O3 -DENABLE_LOGGING src/coparun/runmem.c src/coparun/coparun.c src/coparun/mem_man.c -o bin/coparun-aarch64 