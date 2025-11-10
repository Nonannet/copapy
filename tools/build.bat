python stencils/generate_stencils.py bin/stencils.c

echo -------------x86_64 - 64 bit-----------------
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
echo - Compile stencil test...
cl /Zi /Od stencils\test.c /Fe:bin\test.exe

echo - Build runner for Windows 64 bit...
cl /Zi /Od /DENABLE_BASIC_LOGGING src\coparun\runmem.c src\coparun\coparun.c src\coparun\mem_man.c /Fe:bin\coparun.exe

REM Optimized:
REM cl /O2 src\coparun\runmem.c src\coparun\coparun.c src\coparun\mem_man.c /Fe:bin\coparun.exe

echo - Build stencils for 64 bit...
wsl gcc -fno-pic -c bin/stencils.c -O3 -o src/copapy/obj/stencils_x86_64_O3.o

echo ---------------x86 - 32 bit---------------
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x86

echo - Build runner for Windows 32 bit...
cl /Zi /Od /DENABLE_BASIC_LOGGING src\coparun\runmem.c src\coparun\coparun.c src\coparun\mem_man.c /Fe:bin\coparun-x86.exe

echo - Build runner for linux x86 32 bit...
wsl gcc -m32 -static -Wall -Wextra -Wconversion -Wsign-conversion -Wshadow -Wstrict-overflow -O3 -DENABLE_LOGGING src/coparun/runmem.c src/coparun/coparun.c src/coparun/mem_man.c -o bin/coparun-i686

echo - Build stencils x86 32 bit...
wsl gcc -m32 -fno-pic -c bin/stencils.c -O3 -o src/copapy/obj/stencils_x86_O3.o


echo --------------arm64  64 bit----------------
echo - Build stencils for aarch64...
wsl aarch64-linux-gnu-gcc -fno-pic -c bin/stencils.c -O3 -o src/copapy/obj/stencils_arm64_O3.o

echo ------------------------------
echo - Build runner for Aarch64...
wsl aarch64-linux-gnu-gcc -static -Wall -Wextra -Wconversion -Wsign-conversion -Wshadow -Wstrict-overflow -O3 -DENABLE_LOGGING src/coparun/runmem.c src/coparun/coparun.c src/coparun/mem_man.c -o bin/coparun-aarch64
