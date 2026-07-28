@echo off
title Tesseract Studio Setup
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_tesseract_studio.ps1"
if errorlevel 1 (
  echo.
  echo Setup did not finish. Review the message above, then try again.
  pause
  exit /b 1
)
echo Setup complete. This window will close in five seconds.
timeout /t 5 /nobreak >nul
