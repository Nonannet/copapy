@echo off
REM ============================================================
REM Locate vcvarsall.bat of the newest Visual Studio / Build Tools
REM installation with the C++ toolset and call it.
REM
REM Usage: call tools\vcvars.bat [x64^|x86^|...]
REM Set VCVARSALL to the full path of a vcvarsall.bat to override
REM the automatic lookup.
REM ============================================================

if defined VCVARSALL goto CALL_VCVARS

REM vswhere is always installed at this fixed location (VS 2017 and newer)
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo vswhere.exe not found, set VCVARSALL to the path of vcvarsall.bat
    exit /b 1
)

set "VSINSTALL="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if not defined VSINSTALL (
    echo No Visual Studio installation with C++ tools found, set VCVARSALL to the path of vcvarsall.bat
    exit /b 1
)
set "VCVARSALL=%VSINSTALL%\VC\Auxiliary\Build\vcvarsall.bat"

:CALL_VCVARS
if not exist "%VCVARSALL%" (
    echo vcvarsall.bat not found: "%VCVARSALL%"
    exit /b 1
)
call "%VCVARSALL%" %*
