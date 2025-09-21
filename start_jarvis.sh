#!/bin/bash

# JARVIS Voice Assistant Startup Script
# Starts all components needed for JARVIS integration

echo "🤖 Starting JARVIS Voice Assistant System"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found"
    echo "   Create it with: python -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check API key
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY not set"
    echo "   Set it with: export GEMINI_API_KEY='your-key-here'"
    exit 1
fi

echo "✅ GEMINI_API_KEY found"

# Start JARVIS backend in background
echo "🚀 Starting JARVIS backend server..."
python jarvis_backend.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Test backend health
echo "🏥 Testing backend health..."
if curl -s http://localhost:5001/api/jarvis/health > /dev/null; then
    echo "✅ JARVIS backend is healthy"
else
    echo "❌ JARVIS backend health check failed"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start file server for web interface
echo "🌐 Starting web interface server..."
python serve_jarvis.py &
WEB_PID=$!

# Wait for web server to start
sleep 2

echo ""
echo "🎉 JARVIS System Started Successfully!"
echo "======================================"
echo ""
echo "🎯 Available Interfaces:"
echo "  1. OpenCV AR Application:"
echo "     Run: python main.py"
echo "     Press 'J' to activate JARVIS"
echo ""
echo "  2. Web Interface:"
echo "     Open: http://localhost:8080/jarvis_integration.html"
echo "     Say 'Hey JARVIS' or click to activate"
echo ""
echo "  3. Backend API:"
echo "     Health: http://localhost:5001/api/jarvis/health"
echo "     Status: http://localhost:5001/api/jarvis/status"
echo ""
echo "🗣️  JARVIS Commands:"
echo "  - 'What is this?' - Analyze current view"
echo "  - 'What am I looking at?' - Describe objects"
echo "  - 'Analyze this' - Technical analysis"
echo ""
echo "⌨️  OpenCV Controls:"
echo "  - Press 'J' - Toggle JARVIS"
echo "  - Press '2' - Show 3D objects"
echo "  - Press 'Q' - Quit"
echo ""
echo "🛑 To stop all services:"
echo "   Press Ctrl+C or run: pkill -f jarvis"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down JARVIS system..."
    kill $BACKEND_PID 2>/dev/null
    kill $WEB_PID 2>/dev/null
    echo "✅ JARVIS system stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Keep script running
echo "🤖 JARVIS system running... Press Ctrl+C to stop"
wait
