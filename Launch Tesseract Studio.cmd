@echo off
if not exist "%~dp0.venv-slang\Scripts\python.exe" (
  echo Tesseract Studio needs its one-time setup first.
  echo Opening the setup now...
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_tesseract_studio.ps1"
  if errorlevel 1 (
    echo.
    echo Setup did not finish. Review the message above, then try again.
    pause
    exit /b 1
  )
)
start "" /min powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0run_tesseract_studio.ps1"
