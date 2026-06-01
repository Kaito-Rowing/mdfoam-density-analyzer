@echo off
setlocal

set "PYTHON_EXE=C:\Users\n2002\miniconda3\python.exe"
set "APP_DIR=%~dp0"

if not exist "%PYTHON_EXE%" (
    echo Python not found: %PYTHON_EXE%
    pause
    exit /b 1
)

cd /d "%APP_DIR%"
"%PYTHON_EXE%" app.py

if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)
