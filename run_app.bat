@echo off
title Crop Disease Detection & Environmental Monitoring
echo ============================================================
echo Starting Crop Disease Detection ^& Environmental Monitor...
echo ============================================================
cd /d "%~dp0"
".venv\Scripts\streamlit.exe" run app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with an error.
    pause
)
