@echo off
echo Starting ZeroCode Backend...
cd C:\Users\MOHIT\Desktop\ZeroCode\backend
call conda activate venv
uvicorn app:app --reload --port 8000
pause
