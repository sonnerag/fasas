#!/bin/bash

echo "🚀 Starting Multi-Face Anti-Spoofing Hybrid Application"
echo "======================================================"

# Check if we're in the right directory
if [ ! -f "webcam_multi_face_with_api.py" ]; then
    echo "❌ Error: Please run this script from the Multi-Face-Anti-Spoof-Clear directory"
    exit 1
fi

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        return 0
    else
        return 1
    fi
}

# Check if port 5000 is available
if check_port 5000; then
    echo "⚠️  Port 5000 is already in use. Please stop any existing Flask server first."
    exit 1
fi

echo "✅ Port 5000 is available"

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source venv/bin/activate

# Start the Python backend with API
echo "🚀 Starting Python backend with web API..."
echo "📡 The backend will run the powerful Tkinter detection models"
echo "🌐 API will be available at: http://localhost:5000"
echo "🖥️  Web interface will be available at: http://localhost:5000/web_app_simple.html"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 webcam_multi_face_with_api.py
