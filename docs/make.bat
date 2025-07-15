@ECHO OFF
REM Enhanced batch file for Sphinx documentation
REM Compatible with GitHub Actions and local Windows development
REM Provides comprehensive error handling and debugging output

pushd %~dp0

REM Build configuration variables
if "%SPHINXBUILD%" == "" (
    set SPHINXBUILD=sphinx-build
)
if "%SPHINXAPIDOC%" == "" (
    set SPHINXAPIDOC=sphinx-apidoc
)
set SOURCEDIR=.
set BUILDDIR=_build
set PACKAGE_DIR=..\src\scisbi
set APIDOC_OUTPUT=_autosummary

REM Color support for enhanced output (if supported)
if not defined NO_COLOR (
    set "ESC=["
    set "RESET=[0m"
    set "BOLD=[1m"
    set "RED=[31m"
    set "GREEN=[32m"
    set "YELLOW=[33m"
    set "BLUE=[34m"
) else (
    set "ESC="
    set "RESET="
    set "BOLD="
    set "RED="
    set "GREEN="
    set "YELLOW="
    set "BLUE="
)

REM Check if Sphinx is available
%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
    echo.
    echo %RED%Error: 'sphinx-build' command was not found.%RESET%
    echo.
    echo Make sure you have Sphinx installed:
    echo   pip install sphinx
    echo.
    echo Or set the SPHINXBUILD environment variable to point to the full path
    echo of the 'sphinx-build' executable.
    echo.
    exit /b 1
)

REM Check if sphinx-apidoc is available
%SPHINXAPIDOC% --help >NUL 2>NUL
if errorlevel 9009 (
    echo.
    echo %RED%Error: 'sphinx-apidoc' command was not found.%RESET%
    echo Make sure you have Sphinx installed: pip install sphinx
    echo.
    exit /b 1
)

REM Check if package directory exists
if not exist "%PACKAGE_DIR%" (
    echo.
    echo %RED%Error: Package directory '%PACKAGE_DIR%' not found%RESET%
    echo Please ensure the package structure is correct.
    echo.
    exit /b 1
)

REM Handle command line arguments
if "%1" == "" goto help
if /i "%1" == "help" goto help
if /i "%1" == "clean" goto clean
if /i "%1" == "clean-all" goto clean-all
if /i "%1" == "apidoc" goto apidoc
if /i "%1" == "html" goto html
if /i "%1" == "debug" goto debug
if /i "%1" == "check-deps" goto check-deps
if /i "%1" == "test-build" goto test-build

REM Default Sphinx make mode for other commands
echo %BOLD%Running Sphinx in make mode: %1%RESET%
%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
echo.
echo %BOLD%Sphinx Documentation Build System (Windows)%RESET%
echo.
echo %BOLD%Available targets:%RESET%
echo   %GREEN%help%RESET%        - Show this help message
echo   %GREEN%clean%RESET%       - Remove build files and auto-generated API docs
echo   %GREEN%clean-all%RESET%   - Deep clean including cache and temporary files
echo   %GREEN%apidoc%RESET%      - Generate API documentation from source code
echo   %GREEN%html%RESET%        - Build HTML documentation
echo   %GREEN%debug%RESET%       - Build with maximum verbosity for debugging
echo   %GREEN%check-deps%RESET%  - Check if required dependencies are installed
echo   %GREEN%test-build%RESET%  - Quick test build to check for errors
echo.
echo %BOLD%Environment:%RESET%
echo   SPHINXBUILD   = %SPHINXBUILD%
echo   SPHINXAPIDOC  = %SPHINXAPIDOC%
echo   PACKAGE_DIR   = %PACKAGE_DIR%
echo   BUILDDIR      = %BUILDDIR%
echo.
echo %BOLD%Usage:%RESET%
echo   make.bat [target]
echo.
goto end

:check-deps
echo %BOLD%Checking dependencies...%RESET%
echo %GREEN%✓ All dependencies found%RESET%
goto end

:clean
echo %BOLD%Cleaning documentation build...%RESET%
echo %YELLOW%Removing build directory: %BUILDDIR%%RESET%
if exist "%BUILDDIR%" (
    rmdir /s /q "%BUILDDIR%"
)
echo %YELLOW%Removing auto-generated API documentation...%RESET%
if exist "%APIDOC_OUTPUT%" (
    rmdir /s /q "%APIDOC_OUTPUT%"
)
if exist "scisbi*.rst" (
    del /q scisbi*.rst
)
if exist "modules.rst" (
    del /q modules.rst
)
echo %GREEN%✓ Clean completed%RESET%
goto end

:clean-all
call :clean
echo %BOLD%Performing deep clean...%RESET%
echo %YELLOW%Removing Python cache files...%RESET%
for /d /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d" 2>nul
del /s /q *.pyc 2>nul
del /s /q *.pyo 2>nul
echo %YELLOW%Removing temporary files...%RESET%
del /s /q *~ 2>nul
del /s /q *.tmp 2>nul
echo %GREEN%✓ Deep clean completed%RESET%
goto end

:apidoc
echo %BOLD%Generating API documentation...%RESET%
echo %YELLOW%Running sphinx-apidoc for package: %PACKAGE_DIR%%RESET%
if not exist "%APIDOC_OUTPUT%" (
    mkdir "%APIDOC_OUTPUT%"
)
echo Running: %SPHINXAPIDOC% -f -e -M -o "%APIDOC_OUTPUT%" "%PACKAGE_DIR%" --implicit-namespaces --module-first --separate
%SPHINXAPIDOC% -f -e -M -o "%APIDOC_OUTPUT%" "%PACKAGE_DIR%" --implicit-namespaces --module-first --separate
if errorlevel 1 (
    echo %RED%Error: sphinx-apidoc failed%RESET%
    exit /b 1
)
echo %GREEN%✓ API documentation generated in %APIDOC_OUTPUT%%RESET%
goto end

:html
call :apidoc
echo %BOLD%Building HTML documentation...%RESET%
echo %YELLOW%Source: %SOURCEDIR%%RESET%
echo %YELLOW%Output: %BUILDDIR%\html%RESET%
echo Running: %SPHINXBUILD% -b html "%SOURCEDIR%" "%BUILDDIR%\html" -v -W --keep-going %SPHINXOPTS%
%SPHINXBUILD% -b html "%SOURCEDIR%" "%BUILDDIR%\html" -v -W --keep-going %SPHINXOPTS%
if errorlevel 1 (
    echo %RED%Error: Sphinx build failed%RESET%
    echo %YELLOW%Check the output above for specific error messages%RESET%
    exit /b 1
)
echo %GREEN%✓ HTML documentation built successfully%RESET%
echo %BLUE%Open %BUILDDIR%\html\index.html in your browser%RESET%
goto end

:debug
call :apidoc
echo %BOLD%Building with debug output...%RESET%
echo Running: %SPHINXBUILD% -b html "%SOURCEDIR%" "%BUILDDIR%\html" -v -v -v -T --keep-going %SPHINXOPTS%
%SPHINXBUILD% -b html "%SOURCEDIR%" "%BUILDDIR%\html" -v -v -v -T --keep-going %SPHINXOPTS%
goto end

:test-build
echo %BOLD%Running test build...%RESET%
echo Running: %SPHINXBUILD% -b dummy "%SOURCEDIR%" "%BUILDDIR%\test" -W -q %SPHINXOPTS%
%SPHINXBUILD% -b dummy "%SOURCEDIR%" "%BUILDDIR%\test" -W -q %SPHINXOPTS%
if errorlevel 1 (
    echo %RED%✗ Test build failed%RESET%
    exit /b 1
) else (
    echo %GREEN%✓ Test build passed%RESET%
)
goto end

:end
popd
