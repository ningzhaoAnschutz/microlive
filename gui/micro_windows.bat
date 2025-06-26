@echo off
REM =======================================================
REM launch_gui_windows.bat
REM This batch file activates the "micro" conda environment,
REM changes to the directory containing this file,
REM and launches the GUI script (micro.py).
REM =======================================================

REM Check if conda is available in PATH.
where conda >nul 2>&1
if errorlevel 1 (
    echo Conda does not appear to be installed or is not in your PATH.
    pause
    exit /b 1
)

REM Activate the 'micro' environment.
REM Using "call" is critical in batch files so that the script continues after activation.
call conda activate micro

REM Change to the folder where this batch file resides.
cd /d %~dp0

REM Launch the GUI script.
python micro.py

REM Optional: Pause at the end so the window doesn’t immediately close.
pause
