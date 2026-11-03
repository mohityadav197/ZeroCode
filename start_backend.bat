@echo off
echo Starting ZeroCode Backend...
cd /d "%~dp0backend"
rem The exclude must be an absolute path: uvicorn ignores a relative directory here.
"..\.venv\Scripts\python.exe" -m uvicorn app:app --reload --reload-exclude "%~dp0backend\outputs" --port 8000
pause
