# 🎓 Attendance Management System

A comprehensive attendance management system that combines advanced face recognition with anti-spoofing technology to provide secure, automated attendance tracking.

## ✨ Features

### 🔒 Security Features
- **Anti-Spoofing Detection**: Prevents fake photos, videos, or masks from being used
- **Real-time Face Recognition**: Identifies registered students instantly
- **Always-On Detection**: No need to start/stop - system runs continuously

### 👥 Student Management
- **Easy Registration**: Capture student photos directly with webcam
- **Student Database**: Store student information with face encodings
- **Bulk Management**: Organize students into classes

### 📊 Attendance Tracking
- **Automatic Recording**: Takes attendance when recognized students appear
- **Real-time Updates**: Live display of attendance status
- **Historical Records**: View attendance history by date
- **Confidence Scoring**: Track recognition accuracy

### 🏫 Class Management
- **Class Creation**: Organize students into different classes
- **Flexible Grouping**: Add/remove students from classes
- **Class-based Reports**: Generate attendance reports per class

## 🚀 Quick Start

### 1. Start the System
```bash
./start_attendance_system.sh
```

### 2. Access Web Interface
Open your browser and go to: `http://localhost:5000`

### 3. Register Students
1. Go to "Manage Students" tab
2. Enter Student ID and Name
3. Click "Capture Photo" to take a photo with your webcam
4. Click "Add Student"

### 4. Take Attendance
- The system automatically detects faces and takes attendance
- Students just need to look at the camera
- Real faces are recognized and attendance is recorded
- Fake faces are rejected for security

## 🖥️ Web Interface

### 📹 Live Detection Tab
- Real-time camera feed with face detection
- Live statistics (faces detected, recognized, attendance taken)
- Visual indicators for real/fake faces
- Recognition results with student names

### 👥 Manage Students Tab
- Add new students with photo capture
- View all registered students
- Student database management

### 📊 Attendance Records Tab
- View attendance records by date
- See who attended and when
- Confidence scores for each attendance

### 🏫 Manage Classes Tab
- Create new classes
- Assign students to classes
- View existing classes

## 🔧 Technical Details

### System Requirements
- Python 3.7+
- Webcam/Camera
- Modern web browser
- 4GB+ RAM recommended

### Dependencies
- **Anti-Spoofing Models**: Pre-trained models for liveness detection
- **Face Recognition**: face_recognition library for face matching
- **Web Framework**: Flask for API and web interface
- **Database**: SQLite for data storage
- **Computer Vision**: OpenCV for image processing

### API Endpoints
- `GET /` - Web interface
- `GET /api/health` - System health check
- `GET /api/current_frame` - Live camera feed
- `GET /api/faces` - Current detection results
- `GET /api/students` - List all students
- `POST /api/students` - Add new student
- `GET /api/attendance` - Get attendance records
- `GET /api/classes` - List all classes
- `POST /api/classes` - Create new class

## 🛡️ Security Features

### Anti-Spoofing Protection
- Detects printed photos
- Identifies video replays
- Recognizes 3D masks
- Prevents spoofing attacks

### Recognition Accuracy
- High-confidence face matching
- Multiple model ensemble
- Real-time processing
- Confidence scoring

## 📱 Usage Examples

### For Teachers/Administrators
1. **Setup**: Register all students at the beginning of the semester
2. **Daily Use**: System runs automatically during class
3. **Monitoring**: Check attendance records in real-time
4. **Reports**: Generate attendance reports for grading

### For Students
1. **Registration**: Get photo taken during enrollment
2. **Attendance**: Simply look at the camera when present
3. **Verification**: System confirms attendance automatically

## 🔍 Troubleshooting

### Common Issues
- **Camera not working**: Check camera permissions in browser
- **Low recognition**: Ensure good lighting and clear face view
- **System not starting**: Check if port 5000 is available
- **Database errors**: Delete `attendance.db` to reset

### Performance Tips
- Ensure good lighting for better recognition
- Keep faces centered in camera view
- Avoid multiple people in frame simultaneously
- Use high-resolution camera for better accuracy

## 📈 Future Enhancements

- **Mobile App**: Native mobile application
- **Cloud Sync**: Cloud-based data synchronization
- **Analytics**: Advanced attendance analytics
- **Notifications**: Email/SMS notifications
- **Integration**: LMS integration capabilities

## 🤝 Support

For technical support or feature requests, please check the system logs and ensure all dependencies are properly installed.

---

**Note**: This system is designed for educational environments and provides a secure, automated solution for attendance management using cutting-edge computer vision technology.
