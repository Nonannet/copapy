@echo off
setlocal ENABLEDELAYEDEXPANSION

set ARCH=%1
if "%ARCH%"=="" set ARCH=x86_64

if not "%ARCH%"=="x86_64" ^
if not "%ARCH%"=="x86" ^
if not "%ARCH%"=="arm64" ^
if not "%ARCH%"=="arm-v6" ^
if not "%ARCH%"=="arm-v7" ^
if not "%ARCH%"=="riscv32" ^
if not "%ARCH%"=="riscv64" ^
if not "%ARCH%"=="all" (
    echo Usage: %0 [x86_64^|x86^|arm64^|arm-v6^|arm-v7^|riscv32^|riscv64^|all]
    exit /b 1
)

mkdir build\stencils
mkdir build\runner

python stencils/generate_stencils.py build\stencils\stencils.c

REM ============================================================
REM x86_64
REM ============================================================
if "%ARCH%"=="x86_64" goto BUILD_X86_64
if "%ARCH%"=="all"    goto BUILD_X86_64
goto SKIP_X86_64

:BUILD_X86_64
echo -------------x86_64 - 64 bit-----------------

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

echo - Compile stencil test...
cl /Zi /Od stencils\test.c /Fe:build\stencils\test.exe

echo - Build runner for Windows 64 bit...
cl /Zi /Od /DENABLE_BASIC_LOGGING ^
    src\coparun\runmem.c ^
    src\coparun\coparun.c ^
    src\coparun\mem_man.c ^
    /Fe:build\runner\coparun.exe

echo - Build stencils for x86_64...
wsl gcc -fno-pic -ffunction-sections -c build/stencils/stencils.c -O3 -o build/stencils/stencils.o
wsl ld -r build/stencils/stencils.o build/musl/musl_objects_x86_64.o -o src/copapy/obj/stencils_x86_64_O3.o
wsl objdump -d -x src/copapy/obj/stencils_x86_64_O3.o > build/stencils/stencils_x86_64_O3.asm

:SKIP_X86_64

REM ============================================================
REM x86 32-bit
REM ============================================================
if "%ARCH%"=="x86" goto BUILD_X86
if "%ARCH%"=="all" goto BUILD_X86
goto SKIP_X86

:BUILD_X86
echo ---------------x86 - 32 bit----------------

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x86

echo - Build runner for Windows 32 bit...
cl /Zi /Od /DENABLE_LOGGING ^
    src\coparun\runmem.c ^
    src\coparun\coparun.c ^
    src\coparun\mem_man.c ^
    /Fe:build\runner\coparun-x86.exe

echo - Build runner for Linux x86 32 bit...
wsl i686-linux-gnu-gcc-12 -static -O3 -DENABLE_LOGGING ^
    src/coparun/runmem.c ^
    src/coparun/coparun.c ^
    src/coparun/mem_man.c ^
    -o build/runner/coparun-x86

echo - Build stencils x86 32 bit...
wsl i686-linux-gnu-gcc-12 -fno-pic -ffunction-sections -c build/stencils/stencils.c -O3 -o build/stencils/stencils.o
wsl i686-linux-gnu-ld -r build/stencils/stencils.o build/musl/musl_objects_x86.o -o src/copapy/obj/stencils_x86_O3.o
wsl i686-linux-gnu-objdump -d -x src/copapy/obj/stencils_x86_O3.o > build/stencils/stencils_x86_O3.asm

:SKIP_X86

REM ============================================================
REM ARM64
REM ============================================================
if "%ARCH%"=="arm64" goto BUILD_ARM64
if "%ARCH%"=="all"   goto BUILD_ARM64
goto SKIP_ARM64

:BUILD_ARM64
echo --------------arm64 64 bit----------------

wsl aarch64-linux-gnu-gcc-12 -fno-pic -ffunction-sections -c build/stencils/stencils.c -O3 -o build/stencils/stencils.o
wsl aarch64-linux-gnu-ld -r build/stencils/stencils.o build/musl/musl_objects_arm64.o -o src/copapy/obj/stencils_arm64_O3.o
wsl aarch64-linux-gnu-objdump -d -x src/copapy/obj/stencils_arm64_O3.o > build/stencils/stencils_arm64_O3.asm

echo - Build runner for AArch64...
wsl aarch64-linux-gnu-gcc-12 -static -O3 -DENABLE_LOGGING ^
    src/coparun/runmem.c ^
    src/coparun/coparun.c ^
    src/coparun/mem_man.c ^
    -o build/runner/coparun-aarch64

:SKIP_ARM64

REM ============================================================
REM ARM v6
REM ============================================================
if "%ARCH%"=="arm-v6" goto BUILD_ARMV6
if "%ARCH%"=="all"    goto BUILD_ARMV6
goto SKIP_ARMV6

:BUILD_ARMV6
echo --------------arm-v6 32 bit----------------

wsl arm-none-eabi-gcc -fno-pic -ffunction-sections -march=armv6 -mfpu=vfp -mfloat-abi=hard -marm ^
    -c build/stencils/stencils.c -O3 -o build/stencils/stencils.o

wsl arm-none-eabi-ld -r build/stencils/stencils.o build/musl/musl_objects_armv6.o ^
    $(arm-none-eabi-gcc -print-libgcc-file-name) ^
    -o src/copapy/obj/stencils_armv6_O3.o

wsl arm-none-eabi-objdump -d -x src/copapy/obj/stencils_armv6_O3.o > build/stencils/stencils_armv6_O3.asm

:SKIP_ARMV6

REM ============================================================
REM ARM v7
REM ============================================================
if "%ARCH%"=="arm-v7" goto BUILD_ARMV7
if "%ARCH%"=="all"    goto BUILD_ARMV7
goto SKIP_ARMV7

:BUILD_ARMV7
echo --------------arm-v7 32 bit----------------

wsl arm-none-eabi-gcc -fno-pic -ffunction-sections -march=armv7-a -mfpu=neon-vfpv3 -mfloat-abi=hard -marm ^
    -c build/stencils/stencils.c -O3 -o build/stencils/stencils.o

wsl arm-none-eabi-ld -r build/stencils/stencils.o build/musl/musl_objects_armv7.o ^
    $(arm-none-eabi-gcc -print-libgcc-file-name) ^
    -o src/copapy/obj/stencils_armv7_O3.o

wsl arm-none-eabi-objdump -d -x src/copapy/obj/stencils_armv7_O3.o > build/stencils/stencils_armv7_O3.asm

echo - Build runner for ARM v7...
wsl arm-linux-gnueabihf-gcc -static -O3 -DENABLE_LOGGING ^
    src/coparun/runmem.c ^
    src/coparun/coparun.c ^
    src/coparun/mem_man.c ^
    -o build/runner/coparun-armv7

:SKIP_ARMV7

REM ============================================================
REM RISC-V 64-bit
REM ============================================================
if "%ARCH%"=="riscv64" goto BUILD_RISCV64
if "%ARCH%"=="all"     goto BUILD_RISCV64
goto SKIP_RISCV64

:BUILD_RISCV64
echo -----------riscv64 64 bit-------------

wsl riscv64-linux-gnu-gcc-12 -fno-pic -ffunction-sections -c build/stencils/stencils.c -O3 -o build/stencils/stencils.o
wsl riscv64-linux-gnu-ld -r build/stencils/stencils.o build/musl/musl_objects_riscv64.o -o src/copapy/obj/stencils_riscv64_O3.o
wsl riscv64-linux-gnu-objdump -d -x src/copapy/obj/stencils_riscv64_O3.o > build/stencils/stencils_riscv64_O3.asm

echo - Build runner for RISC-V 64 bit...
wsl riscv64-linux-gnu-gcc-12 -static -O3 -DENABLE_LOGGING ^
    src/coparun/runmem.c ^
    src/coparun/coparun.c ^
    src/coparun/mem_man.c ^
    -o build/runner/coparun-riscv64

:SKIP_RISCV64

REM ============================================================
REM RISC-V 32-bit
REM ============================================================
if "%ARCH%"=="riscv32" goto BUILD_RISCV32
if "%ARCH%"=="all"     goto BUILD_RISCV32
goto SKIP_RISCV32

:BUILD_RISCV32
echo -----------riscv32 32 bit-------------
REM cross compiler: https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2026.04.26/riscv32-musl-ubuntu-22.04-gcc.tar.xz

wsl build/riscv_gcc/riscv/bin/riscv32-unknown-linux-musl-gcc -fno-pic -ffunction-sections -c build/stencils/stencils.c -O3 -o build/stencils/stencils.o
wsl build/riscv_gcc/riscv/bin/riscv32-unknown-linux-musl-ld -r build/stencils/stencils.o build/musl/musl_objects_riscv32.o -o src/copapy/obj/stencils_riscv32_O3.o
wsl build/riscv_gcc/riscv/bin/riscv32-unknown-linux-musl-objdump -d -x src/copapy/obj/stencils_riscv32_O3.o > build/stencils/stencils_riscv32_O3.asm

echo - Build runner for RISC-V 32 bit...
wsl build/riscv_gcc/riscv/bin/riscv32-unknown-linux-musl-gcc -static -O3 -DENABLE_LOGGING ^
    src/coparun/runmem.c ^
    src/coparun/coparun.c ^
    src/coparun/mem_man.c ^
    -o build/runner/coparun-riscv32

:SKIP_RISCV32

:END
echo Build completed for %ARCH%
endlocal
