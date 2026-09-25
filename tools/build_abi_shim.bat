@echo off
setlocal

REM ============================================================
REM Build a stripped x86_64 ABI shim object, then emit an annotated
REM hex dump. The published .obj is the stripped one (debug + local
REM symbols removed via objcopy --strip-debug --strip-unneeded).
REM ============================================================

mkdir build\runner

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

echo - Build entry wrapper (raw)...
ml64 /c /Fobuild\runner\x86_64_abi_shim.raw.obj src\coparun\x86_64_abi_shim.asm
if errorlevel 1 (
    echo Assembler failed.
    exit /b 1
)

echo - Strip debug info into final object...
wsl objcopy --strip-debug --strip-unneeded build/runner/x86_64_abi_shim.raw.obj build/runner/x86_64_abi_shim.obj
if errorlevel 1 (
    echo objcopy failed; keeping unstripped object.
    copy /Y build\runner\x86_64_abi_shim.raw.obj build\runner\x86_64_abi_shim.obj >nul
)
del build\runner\x86_64_abi_shim.raw.obj

echo - Annotated hex dump of entry wrapper...
set "PY=python"
if exist .venv\Scripts\python.exe set "PY=.venv\Scripts\python.exe"
"%PY%" tools\hexdump_annotated.py build\runner\x86_64_abi_shim.obj src\coparun\x86_64_abi_shim.hex
if errorlevel 1 (
    echo Hex dump failed.
    exit /b 1
)

echo Annotated hex dump written to src\coparun\x86_64_abi_shim.hex
endlocal
