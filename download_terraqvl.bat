@echo off
echo ====================================================
echo  SATQUERY AI - TerraQ-VL Model Downloader
echo  PS 26167 / ISRO Remote Sensing AI
echo ====================================================
echo.
echo This script downloads the TerraQ-VL local model (~14 GB).
echo You only need to do this ONCE.
echo After completion, start_app.bat will automatically
echo enable TerraQ-VL local inference.
echo.
echo Requirements:
echo  - Stable internet connection
echo  - At least 20 GB free disk space
echo  - Dependencies installed (run install.bat first)
echo.
pause
echo.
echo Starting model download...
echo This may take 30 minutes to several hours depending on
echo your internet speed. Do NOT close this window.
echo.
python scripts\download_vlm.py
if %errorlevel% equ 0 (
    echo.
    echo ====================================================
    echo  SUCCESS: TerraQ-VL model downloaded!
    echo  Restart start_app.bat to enable local AI inference.
    echo ====================================================
) else (
    echo.
    echo ====================================================
    echo  ERROR: Download failed.
    echo  Check your internet connection and try again.
    echo  Make sure install.bat was run first.
    echo ====================================================
)
echo.
pause
