@echo off
echo Starting NBA Shot Pattern Explorer Frontend...
echo.

cd frontend

if not exist node_modules (
    echo Installing dependencies...
    call npm install
)

echo.
echo Starting React development server on http://localhost:3000
echo.

call npm start
