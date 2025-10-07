call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

REM With Symbols:
cl /Zi /Od src\coparun\runmem.c src\coparun\coparun.c src\coparun\mem_man.c /Fe:bin\coparun.exe

REM Optimized:
REM cl /O2 src\coparun\runmem.c src\coparun\coparun.c src\coparun\mem_man.c /Fe:bin\coparun.exe