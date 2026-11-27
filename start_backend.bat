@echo off
title SAHAAY AI Backend Server
echo Starting FastAPI Backend at http://localhost:8000 ...
cd /d %~dp0backend
call ..\.venv\Scripts\activate.bat
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
