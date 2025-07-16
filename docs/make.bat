@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto help
if "%1" == "help" goto help
if "%1" == "clean" goto clean
if "%1" == "html" goto html
if "%1" == "check-deps" goto check-deps

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 1 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable.
	echo.
	exit /b 1
)

goto end

:help
echo.Sphinx documentation build targets:
echo.  html       Build HTML documentation
echo.  clean      Remove build files
echo.  check-deps Check dependencies
goto end

:clean
rmdir /s /q "%BUILDDIR%" > nul 2>&1
echo.Build files removed.
goto end

:html
%SPHINXBUILD% -b html %SOURCEDIR% %BUILDDIR%\html
if errorlevel 1 exit /b 1
echo.
echo.Build finished. The HTML pages are in %BUILDDIR%\html.
goto end

:check-deps
python -c "import sphinx, furo, myst_parser; print('All dependencies available')"
goto end

:end
popd
