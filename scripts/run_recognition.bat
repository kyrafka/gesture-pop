@echo off
set "ROOT=%~dp0.."
cd /d "%ROOT%"

if exist "%ROOT%\.venv\Scripts\python.exe" (
  "%ROOT%\.venv\Scripts\python.exe" "%ROOT%\gesture_launcher.py"
) else (
  py -3.12 "%ROOT%\gesture_launcher.py"
)

if errorlevel 1 pause
