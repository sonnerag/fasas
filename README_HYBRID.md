# ��️ Multi-Face Anti-Spoofing - Hybrid Web Application

This is a powerful hybrid solution that combines the best of both worlds:
- **Powerful Python backend** with the same accurate models as the Tkinter desktop app
- **Modern web interface** for easy access and control

## 🚀 Quick Start

### 1. Start the Application
```bash
./start_hybrid_app.sh
```

Or manually:
```bash
source venv/bin/activate
python3 webcam_multi_face_with_api.py
```

### 2. Access the Web Interface
Open your browser and go to:
- **Main interface**: http://localhost:5000
- **Direct link**: http://localhost:5000/web_app_simple.html

## 🎯 Features

### Backend (Python)
- ✅ **Same powerful models** as the Tkinter desktop version
- ✅ **Real-time detection** with up to 5 faces simultaneously
- ✅ **High accuracy** anti-spoofing detection
- ✅ **REST API** for web interface communication
- ✅ **Live camera feed** processing

### Frontend (Web Interface)
- ✅ **Real-time video feed** from your camera
- ✅ **Live face detection boxes** with proper scaling
- ✅ **Detection statistics** (total, real, fake faces, FPS)
- ✅ **Interactive controls** (max faces, threshold)
- ✅ **Start/Stop detection** buttons
- ✅ **Responsive design** works on desktop and mobile

## 🎛️ Controls

### Detection Settings
- **Max Faces**: Set how many faces to detect (1-10)
- **Threshold**: Adjust detection sensitivity (0.1-1.0)
- **Start/Stop**: Control detection in real-time

### Real-time Stats
- **Total Faces**: Number of faces currently detected
- **Real Faces**: Number of genuine faces
- **Fake Faces**: Number of spoofed/fake faces
- **FPS**: Current processing speed

## 🔧 API Endpoints

The backend provides these REST API endpoints:

- `GET /api/health` - Check backend status
- `GET /api/current_frame` - Get current camera frame
- `GET /api/faces` - Get current detection results
- `POST /api/start_detection` - Start detection
- `POST /api/stop_detection` - Stop detection
- `GET /api/settings` - Get current settings
- `POST /api/settings` - Update settings

## 🎨 Face Detection Display

- **Green boxes**: Real/genuine faces
- **Red boxes**: Fake/spoofed faces
- **Confidence percentage**: Shows detection confidence
- **Proper scaling**: Boxes automatically adjust to screen size

## 🛠️ Technical Details

### Backend Architecture
- **Flask API server** running on port 5000
- **Multi-threaded detection** for smooth performance
- **OpenCV camera capture** with optimized settings
- **Same anti-spoofing models** as desktop version

### Frontend Architecture
- **Pure HTML/CSS/JavaScript** (no React complexity)
- **Real-time updates** via REST API calls
- **Responsive design** with CSS Grid
- **Automatic scaling** for face detection boxes

## 🔍 Troubleshooting

### Backend Issues
- **Port 5000 in use**: Stop other Flask servers or change port
- **Camera not found**: Check camera permissions and connections
- **Models not loading**: Ensure model files are in correct directory

### Frontend Issues
- **Boxes not aligned**: The scaling has been fixed in the latest version
- **Connection failed**: Check if backend is running on port 5000
- **No video feed**: Ensure camera is not being used by other applications

## 🎯 Performance

This hybrid solution provides:
- **Same accuracy** as the Tkinter desktop version
- **Real-time processing** at ~30 FPS
- **Low latency** web interface updates
- **Efficient resource usage** with optimized threading

## 🌟 Advantages Over Pure Web Solutions

1. **Accuracy**: Uses the same trained models as desktop version
2. **Performance**: Native Python processing vs browser limitations
3. **Reliability**: Proven detection algorithms
4. **Flexibility**: Easy to modify and extend
5. **Compatibility**: Works with any modern web browser

Enjoy your powerful multi-face anti-spoofing web application! 🎉
