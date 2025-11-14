mkdir build\stencils
mkdir build\runner
python stencils/generate_stencils.py build/stencils/stencils.c

echo -------------x86_64 - 64 bit-----------------
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
echo - Compile stencil test...
cl /Zi /Od stencils\test.c /Fe:build\stencils\test.exe

echo - Build runner for Windows 64 bit...
cl /Zi /Od /DENABLE_BASIC_LOGGING src\coparun\runmem.c src\coparun\coparun.c src\coparun\mem_man.c /Fe:build\runner\coparun.exe

REM Optimized:
REM cl /O2 src\coparun\runmem.c src\coparun\coparun.c src\coparun\mem_man.c /Fe:build\runner\coparun.exe

echo - Build stencils for 64 bit...
REM ../copapy/tools/cross_compiler_unix/packobjs.sh gcc ld ../copapy/build/musl/musl_objects_x86_64.o
wsl gcc -fno-pic -ffunction-sections -c build/stencils/stencils.c -O3 -o build/stencils/stencils.o
wsl ld -r build/stencils/stencils.o build/musl/musl_objects_x86_64.o -o src/copapy/obj/stencils_x86_64_O3.o
wsl objdump -d -x src/copapy/obj/stencils_x86_64_O3.o > build/stencils/stencils_x86_64_O3.asm

echo ---------------x86 - 32 bit---------------
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x86

echo - Build runner for Windows 32 bit...
cl /Zi /Od /DENABLE_LOGGING src\coparun\runmem.c src\coparun\coparun.c src\coparun\mem_man.c /Fe:build\runner\coparun-x86.exe

echo - Build runner for linux x86 32 bit...
wsl i686-linux-gnu-gcc-12 -static -Wall -Wextra -Wconversion -Wsign-conversion -Wshadow -Wstrict-overflow -O3 -DENABLE_LOGGING src/coparun/runmem.c src/coparun/coparun.c src/coparun/mem_man.c -o build/runner/coparun-x86

echo - Build stencils x86 32 bit...
REM  sh ../copapy/tools/cross_compiler_unix/packobjs.sh i686-linux-gnu-gcc-12 i686-linux-gnu-ld ../copapy/build/musl/musl_objects_x86.o
wsl i686-linux-gnu-gcc-12 -fno-pic -ffunction-sections -c build/stencils/stencils.c -O3 -o build/stencils/stencils.o
wsl i686-linux-gnu-ld -r build/stencils/stencils.o build/musl/musl_objects_x86.o -o src/copapy/obj/stencils_x86_O3.o
wsl i686-linux-gnu-objdump -d -x src/copapy/obj/stencils_x86_O3.o > build/stencils/stencils_x86_O3.asm


echo --------------arm64  64 bit----------------
echo - Build stencils for aarch64...
wsl aarch64-linux-gnu-gcc-12 -fno-pic -ffunction-sections -c build/stencils/stencils.c -O3 -o build/stencils/stencils.o
wsl aarch64-linux-gnu-ld -r build/stencils/stencils.o build/musl/musl_objects_arm64.o -o src/copapy/obj/stencils_arm64_O3.o
wsl aarch64-linux-gnu-objdump -d -x src/copapy/obj/stencils_arm64_O3.o > build/stencils/stencils_arm64_O3.asm

echo ------------------------------
echo - Build runner for Aarch64...
wsl aarch64-linux-gnu-gcc-12 -static -Wall -Wextra -Wconversion -Wsign-conversion -Wshadow -Wstrict-overflow -O3 -DENABLE_LOGGING src/coparun/runmem.c src/coparun/coparun.c src/coparun/mem_man.c -o build/runner/coparun-aarch64
