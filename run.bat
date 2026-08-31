@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo   Starting Fraud Red-Team Simulation Dashboard...
echo ===================================================

cd /d "%~dp0"

:: Load .env file if it exists to check for GROQ_API_KEY
if exist .env (
    echo -^> Loading API Keys from .env file...
    for /f "usebackq tokens=1,* delims==" %%A in (.env) do (
        if "%%A"=="GROQ_API_KEY" (
            :: Remove quotes if they exist
            set GROQ_API_KEY=%%~B
        )
    )
)

:: Prompt for API key if missing
if "!GROQ_API_KEY!"=="" (
    echo.
    echo ===================================================
    echo WARNING: GROQ_API_KEY is not set!
    echo The Red Team simulator requires a Groq API Key to invent scenarios.
    echo You can get a free key instantly at: https://console.groq.com/keys
    echo ===================================================
    set /p USER_API_KEY="Please enter your Groq API Key: "
    
    if "!USER_API_KEY!"=="" (
        echo Error: API Key cannot be empty. Exiting.
        pause
        exit /b 1
    )
    
    echo GROQ_API_KEY="!USER_API_KEY!" >> .env
    set GROQ_API_KEY=!USER_API_KEY!
    echo -^> Saved your key to the .env file successfully!
    echo.
)

echo -^> Starting Backend API (Port 8000)...
cd siem-dashboard\backend

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

:: Start backend in a new minimized window
start "PAHREDAAR Backend" /MIN cmd /c "uvicorn main:app --reload --host 0.0.0.0 --port 8000 > backend.log 2>&1"

echo -^> Starting Frontend Dashboard (Port 5173)...
cd ..\frontend

:: Start frontend in a new minimized window
start "PAHREDAAR Frontend" /MIN cmd /c "npm run dev -- --host 0.0.0.0 --port 5173 > frontend.log 2>&1"

echo.
echo ===================================================
echo   All Systems Go!
echo   Access the dashboard here: http://localhost:5173
echo.
echo   Note: The backend and frontend are running in 
echo   separate background/minimized windows.
echo   To stop the servers, just close those windows.
echo ===================================================
pause
