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
echo Compiling Sovereign_Fractal_Core.cpp...
cl.exe /std:c++17 /EHsc /O2 Sovereign_Fractal_Core.cpp
if %errorlevel% neq 0 (
    echo Compilation failed.
    exit /b %errorlevel%
)
echo.
echo === RUNNING PHASE 14 FRACTAL ===
Sovereign_Fractal_Core.exe
