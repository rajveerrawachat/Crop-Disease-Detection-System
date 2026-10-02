@echo off
title Crop Disease Detection - Test Pipeline
echo ============================================================
echo Running Automated Test Pipeline...
echo ============================================================
cd /d "%~dp0"
".venv\Scripts\python.exe" test_pipeline.py
echo.
pause
