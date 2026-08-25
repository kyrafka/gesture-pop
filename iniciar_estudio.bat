@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" gesture_studio.py
) else (
  python gesture_studio.py
)
if errorlevel 1 pause
