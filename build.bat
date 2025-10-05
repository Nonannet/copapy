call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cl /O2 src\coparun\runmem.c src\coparun\coparun.c src\coparun\mem_man.c /Fe:bin\coparun.exe