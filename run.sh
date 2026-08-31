#!/bin/bash

echo "==================================================="
echo "  Starting Fraud Red-Team Simulation Dashboard..."
echo "==================================================="

# Navigate to the project root just in case
cd "$(dirname "$0")"

# Load the .env file automatically so you don't need VS Code's terminal injection
if [ -f .env ]; then
    echo "-> Loading API Keys from .env file..."
    export $(grep -v '^#' .env | xargs) 2>/dev/null
fi

if [ -z "$GROQ_API_KEY" ]; then
    echo ""
    echo "==================================================="
    echo "⚠️  GROQ_API_KEY is not set!"
    echo "The Red Team simulator requires a Groq API Key to invent scenarios."
    echo "You can get a free key instantly at: https://console.groq.com/keys"
    echo "==================================================="
    read -p "Please enter your Groq API Key: " user_api_key
    if [ -z "$user_api_key" ]; then
        echo "Error: API Key cannot be empty. Exiting."
        exit 1
    fi
    echo "export GROQ_API_KEY=\"$user_api_key\"" >> .env
    export GROQ_API_KEY="$user_api_key"
    echo "-> Saved your key to the .env file successfully!"
    echo ""
fi

# Start the FastAPI backend
echo "-> Starting Backend API (Port 8000)..."
cd siem-dashboard/backend

# Activate virtual environment if it exists (common for Python projects)
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the backend in the background
uvicorn main:app --reload --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!
echo "   Backend running with PID: $BACKEND_PID"

# Start the Vite frontend
echo "-> Starting Frontend Dashboard (Port 5173)..."
cd ../frontend

# Run the frontend in the background
npm run dev -- --host 0.0.0.0 --port 5173 > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   Frontend running with PID: $FRONTEND_PID"

echo ""
echo "==================================================="
echo "  🚀 All Systems Go!"
echo "  Access the dashboard here: http://localhost:5173"
echo "  Press [Ctrl+C] to stop both servers safely."
echo "==================================================="

# Setup trap to catch Ctrl+C and kill both background processes
trap "echo -e '\nStopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Goodbye!'; exit 0" SIGINT SIGTERM

# Keep script running and wait for background processes
wait $BACKEND_PID $FRONTEND_PID
