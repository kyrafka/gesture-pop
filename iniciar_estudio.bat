@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" gesture_studio_qt.py
) else (
  py -3.12 gesture_studio_qt.py
)
if errorlevel 1 pause
