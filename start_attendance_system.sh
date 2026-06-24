#!/bin/bash

echo "🎓 Starting Attendance Management System"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "attendance_system.py" ]; then
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

# Check if port 8000 is available
if check_port 8000; then
    echo "⚠️  Port 8000 is already in use. Please stop any existing server first."
    exit 1
fi

echo "✅ Port 8000 is available"

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source venv/bin/activate

# Install additional dependencies if needed (dlib precompiled to avoid building from source)
echo "📦 Checking dependencies..."
pip install -q flask flask-cors dlib-bin
pip install -q --no-deps face_recognition face_recognition_models

# Start the Attendance Management System
echo "🚀 Starting Attendance Management System..."
echo "📡 The system will run with always-on detection"
echo "🌐 Web interface will be available at: http://localhost:8000"
echo "🎯 Features:"
echo "   - Live face detection with anti-spoofing"
echo "   - Student management with photo capture"
echo "   - Automatic attendance tracking"
echo "   - Class management"
echo "   - Attendance records and reports"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 attendance_system.py
