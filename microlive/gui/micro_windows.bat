@echo off
REM =======================================================
REM MicroLive GUI Launcher (Windows)
REM This batch file activates the conda environment
REM and launches MicroLive using the pip-installed entry point.
REM =======================================================

REM Check if conda is available in PATH.
where conda >nul 2>&1
if errorlevel 1 (
    echo Conda does not appear to be installed or is not in your PATH.
    pause
    exit /b 1
)

REM Activate the 'microlive' environment.
REM Using "call" is critical in batch files so that the script continues after activation.
call conda activate microlive

REM Launch MicroLive using the pip-installed entry point
microlive

REM Optional: Pause at the end so the window doesn't immediately close.
pause
