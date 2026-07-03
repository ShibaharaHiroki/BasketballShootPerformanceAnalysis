@echo off
echo Starting Basketball Shoot Performance Analysis Backend...
echo.

cd backend

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Installing dependencies...
.\venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo Starting FastAPI server on http://localhost:8000
echo API documentation available at http://localhost:8000/docs
echo.

.\venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
