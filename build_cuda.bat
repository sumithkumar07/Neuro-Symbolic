@echo off
set "VCVARS="
for /f "tokens=*" %%i in ('dir /b /s "C:\Program Files\Microsoft Visual Studio\*vcvarsall.bat" 2^>nul') do (
    set "VCVARS=%%i"
    goto :found
)
for /f "tokens=*" %%i in ('dir /b /s "C:\Program Files (x86)\Microsoft Visual Studio\*vcvarsall.bat" 2^>nul') do (
    set "VCVARS=%%i"
    goto :found
)
:found
if "%VCVARS%"=="" (
    echo Visual Studio compiler not found.
    exit /b 1
)
call "%VCVARS%" x64
echo Building Neuro-Symbolic GPU Core...
nvcc neuro_symbolic_core.cu -o bin\neuro_symbolic.exe -O3 -allow-unsupported-compiler -std=c++17
if %errorlevel% neq 0 (
    echo Compilation failed.
    exit /b %errorlevel%
)
echo Build Completed.
echo.
echo === RUNNING NEURO-SYMBOLIC CORE ===
bin\neuro_symbolic.exe
