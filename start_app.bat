@echo off
echo ====================================================
echo  SATQUERY AI  ^|  PS 26167  ^|  ISRO
echo  Vision-Language Assistant for Remote Sensing
echo ====================================================
echo.

:: Check TerraQ-VL model status
if exist vlm_ready.flag (
    echo  [READY]   TerraQ-VL Local Model  - Active
) else (
    echo  [STANDBY] TerraQ-VL Local Model  - Run download_terraqvl.bat to enable
)
echo  [ACTIVE]  Gemini 3.5 Flash        - Cloud Fallback
echo  [ACTIVE]  Backend API Server      - Starting...
echo  [ACTIVE]  React Frontend          - Served at /
echo.
echo  Open in browser: http://localhost:5000
echo  Press CTRL+C to stop the server
echo ====================================================
echo.

:: Open browser after 3 seconds (non-blocking)
start /b cmd /c "timeout /t 3 /nobreak > nul & start http://localhost:5000"

:: Start the server in this window (blocking - server stays open)
python api_app.py
