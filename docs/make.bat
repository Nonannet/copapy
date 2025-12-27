@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help

md %BUILDDIR%

python %SOURCEDIR%\generate_class_list.py --api-dir %SOURCEDIR%\api
python %SOURCEDIR%\extract_section.py --readme %SOURCEDIR%/../../README.md --build-dir %BUILDDIR%
python %SOURCEDIR%\stencil_doc.py --input "%SOURCEDIR%/../../build/stencils.c" --asm-pattern "%SOURCEDIR%/../../build/tmp/runner-linux-*/stencils.asm" --output %BUILDDIR%/stencils.md
python %SOURCEDIR%\example_asm.py --input "%SOURCEDIR%/../../tools/make_example.py" --asm-pattern "%SOURCEDIR%/../../build/tmp/runner-linux-*/example.asm" --output %BUILDDIR%/compiled_example.md

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
