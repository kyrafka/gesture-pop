@echo off
set "ROOT=%~dp0.."
cd /d "%ROOT%"

if exist "%ROOT%\.venv\Scripts\pythonw.exe" (
  start "" "%ROOT%\.venv\Scripts\pythonw.exe" "%ROOT%\gesture_studio_qt.py"
  exit /b 0
)

py -3.12 "%ROOT%\gesture_studio_qt.py"
if errorlevel 1 pause
