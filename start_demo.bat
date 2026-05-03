@echo off
echo ========================================
echo   AQT Data Intelligence - Demo Startup
echo ========================================
echo.

:: Start backend in a new window
echo [1/3] Starting Backend API...
start "AQT Backend" cmd /k "cd /d %~dp0 && uvicorn main:app --host 0.0.0.0 --port 8000"

:: Wait for backend to start
timeout /t 3 /nobreak >nul

:: Start frontend in a new window
echo [2/3] Starting Frontend...
start "AQT Frontend" cmd /k "cd /d %~dp0Frontend_Data && npm run dev"

:: Start ngrok in a new window
echo [3/3] Starting ngrok tunnel...
start "AQT ngrok" cmd /k "ngrok http 8000"

echo.
echo ========================================
echo   All services starting...
echo   Frontend: http://localhost:5173
echo   Backend:  http://localhost:8000
echo   ngrok:    Check the ngrok window for public URL
echo ========================================
echo.
pause
