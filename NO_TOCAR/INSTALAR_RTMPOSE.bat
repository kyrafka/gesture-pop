@echo off
set "ROOT=%~dp0.."
cd /d "%ROOT%"

if not exist "%ROOT%\.venv\Scripts\python.exe" (
  echo No encontre .venv. Crea el entorno Python 3.12 primero.
  pause
  exit /b 1
)

"%ROOT%\.venv\Scripts\python.exe" -m pip install -r "%ROOT%\requirements-heavy.txt"
if errorlevel 1 goto error
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\setup_heavy_assist.py"
if errorlevel 1 goto error

echo.
echo RTMPose quedo listo. Ya puedes abrir Gesture Pop.
pause
exit /b 0

:error
echo.
echo No pude completar la instalacion de RTMPose.
pause
exit /b 1
