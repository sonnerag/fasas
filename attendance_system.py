#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attendance Management System with Multi-Face Anti-Spoofing
Combines face recognition with anti-spoofing for secure attendance tracking
"""

import cv2
import numpy as np
import time
import os
import sys
import json
import threading
import sqlite3
import base64
from datetime import datetime, date, time as dt_time, timedelta
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
from io import BytesIO
from PIL import Image
import face_recognition
import pickle

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.src.anti_spoof_predict import AntiSpoofPredict
from core.src.generate_patches import CropImage
from core.src.utility import parse_model_name
from config.settings import *

class AttendanceSystem:
    def __init__(self, model_dir=None, device_id=None, max_faces=None):
        # Use default values from config if not provided
        self.model_dir = model_dir or MODEL_DIR
        self.device_id = device_id if device_id is not None else DEFAULT_DEVICE_ID
        self.max_faces = max_faces or DEFAULT_MAX_FACES
        self.threshold = DEFAULT_SPOOF_THRESHOLD
        
        # Initialize components
        self.model_test = AntiSpoofPredict(self.device_id)
        self.image_cropper = CropImage()
        
        # Load models
        self.models = []
        for model_name in os.listdir(self.model_dir):
            if model_name.endswith('.pth'):
                self.models.append(model_name)
        
        print(f"Loaded {len(self.models)} models: {self.models}")
        print(f"Max faces to detect: {self.max_faces}")
        print(f"Using device: {'GPU' if self.device_id > 0 else 'CPU'}")
        
        # Camera setup - start with no camera
        self.cap = None
        self.camera_initialized = False
        
        # Initialize database
        self.init_database()
        
        self.recognition_lock = threading.Lock()
        self.dlib_lock = threading.Lock()
        
        # Load face encodings
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.load_face_encodings()
        
        # Detection state
        self.running = True
        self.detection_active = False  # Start with detection off
        self.current_frame = None
        self.current_faces = []
        self.detection_stats = {
            'total_faces': 0,
            'real_faces': 0,
            'fake_faces': 0,
            'recognized_faces': 0,
            'attendance_taken': 0,
            'fps': 0
        }
        
        # ── Performance: recognition runs in its own slow thread ──────────────
        # Cache: maps face-index → {recognizedName, recognizedId, classInfo}
        self.recognition_cache = {}
        # Frame queued for recognition thread to process
        self.recognition_frame_queue = None
        self.recognition_bbox_queue = []   # list of (bbox, face_index) from latest spoof pass
        self.recognition_interval = 1.0    # seconds between recognition passes
        # ── Anti-spoof temporal gate ──────────────────────────────────────────
        # A recognized student must be confirmed REAL on this many CONSECUTIVE
        # recognition cycles before attendance is taken. A photo/phone usually
        # fools the liveness model for only the first instant; by the time the
        # streak completes the model has flipped it to "fake", the face drops out
        # of real_bboxes, the streak resets, and no attendance is recorded.
        self.required_real_cycles = 3
        self.real_streak = {}              # student_id → consecutive real+recognized cycles
        # ─────────────────────────────────────────────────────────────────────
        
        # Attendance tracking
        self.attendance_today = set()  # Track who has already taken attendance today
        self.last_attendance_time = {}  # Track last attendance time per student
        self.recent_attendance_messages = []  # Store recent attendance messages
        
        # Flask app setup
        self.app = Flask(__name__)
        CORS(self.app)
        self.setup_api_routes()
        
        # Start detection thread (but not active by default)
        self.detection_thread = threading.Thread(target=self.detection_loop, name='DetectionLoop')
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        # Start face-recognition thread (slow, decoupled from spoof detection)
        self.recognition_thread = threading.Thread(target=self.recognition_loop, name='RecognitionLoop')
        self.recognition_thread.daemon = True
        self.recognition_thread.start()
        
        # Camera streaming lock and background thread
        self.camera_lock = threading.Lock()
        self.camera_reader_thread = None
        
        print("✅ Attendance system started - detection is OFF by default")

    def init_camera(self):
        """Initialize camera for detection"""
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(DEFAULT_CAMERA_ID)
        if not self.cap.isOpened():
            print(f"Could not open camera {DEFAULT_CAMERA_ID}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.camera_initialized = True
        
        # Start background camera reader thread to empty buffer and keep stream lag-free
        if self.camera_reader_thread is None or not self.camera_reader_thread.is_alive():
            self.camera_reader_thread = threading.Thread(target=self.camera_loop, name='CameraReaderLoop')
            self.camera_reader_thread.daemon = True
            self.camera_reader_thread.start()
            print("🎥 Camera reader thread started")

        print(f"Camera {DEFAULT_CAMERA_ID} initialized successfully")
        return True

    def camera_loop(self):
        """Continuously reads frames from the camera to avoid any backlog in the OpenCV buffer."""
        while self.running:
            # Read and release must be mutually exclusive: on macOS (AVFoundation)
            # calling cap.release() while another thread is inside cap.read() frees the
            # CaptureDelegate mid-grab and crashes with a segfault. Holding camera_lock
            # around read() guarantees release_camera() never runs during a read.
            with self.camera_lock:
                cap = self.cap
                if self.camera_initialized and cap is not None:
                    ret, frame = cap.read()
                else:
                    ret, frame = False, None
                if ret:
                    self.current_frame = frame
            if not ret:
                time.sleep(0.03)

    def release_camera(self):
        """Release camera resources"""
        # Take camera_lock so we never release while camera_loop is inside cap.read()
        # (see camera_loop) — that race segfaults on macOS.
        with self.camera_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                self.camera_initialized = False
                print("Camera released")

    def init_database(self):
        """Initialize SQLite database for students and attendance"""
        self.db_path = "attendance.db"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create students table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                face_encoding BLOB,
                photo_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create attendance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                date DATE NOT NULL,
                time TIME NOT NULL,
                status TEXT DEFAULT 'present',
                class_name TEXT,
                FOREIGN KEY (student_id) REFERENCES students (student_id)
            )
        ''')
        
        # Create classes table with scheduling
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS classes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_name TEXT NOT NULL,
                student_ids TEXT,  -- JSON array of student IDs
                class_date DATE NOT NULL,
                class_time TIME NOT NULL,
                duration_minutes INTEGER DEFAULT 60,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized")

    def load_face_encodings(self):
        """Load face encodings from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT student_id, name, face_encoding FROM students WHERE face_encoding IS NOT NULL')
        students = cursor.fetchall()
        
        new_encodings = []
        new_names = []
        new_ids = []
        
        for student_id, name, face_encoding_blob in students:
            try:
                face_encoding = pickle.loads(face_encoding_blob)
                new_encodings.append(face_encoding)
                new_names.append(name)
                new_ids.append(student_id)
            except Exception as e:
                print(f"Error loading face encoding for {name}: {e}")
        
        with self.recognition_lock:
            self.known_face_encodings = new_encodings
            self.known_face_names = new_names
            self.known_face_ids = new_ids
        
        conn.close()
        print(f"✅ Loaded {len(self.known_face_encodings)} face encodings")

    def normalize_time(self, time_str):
        """Normalize time string to HH:MM:SS format"""
        if not time_str:
            return None
        
        # Handle different time formats
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 2:  # HH:MM
                return f"{parts[0]}:{parts[1]}:00"
            elif len(parts) == 3:  # HH:MM:SS
                return time_str
        return None

    def get_student_class_info(self, student_id):
        """Get the class information for a student"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current date and time
        now = datetime.now()
        current_date = now.date()
        current_time = now.time()
        
        print(f"🔍 Checking class for student {student_id} at {current_date} {current_time}")
        
        # Find active class for this student
        cursor.execute('''
            SELECT class_name, class_date, class_time, duration_minutes
            FROM classes 
            WHERE student_ids LIKE ? 
            AND class_date = ?
        ''', (f'%"{student_id}"%', current_date))
        
        classes = cursor.fetchall()
        conn.close()
        
        for class_name, class_date, class_time, duration in classes:
            # Normalize time formats
            normalized_class_time = self.normalize_time(class_time)
            if not normalized_class_time:
                continue
                
            # Parse class time
            try:
                class_time_obj = datetime.strptime(normalized_class_time, '%H:%M:%S').time()
                
                # Calculate class end time
                class_start = datetime.combine(current_date, class_time_obj)
                class_end = class_start + timedelta(minutes=duration)
                class_end_time = class_end.time()
                
                print(f"📅 Class: {class_name}, Start: {class_time_obj}, End: {class_end_time}, Current: {current_time}")
                
                # Check if current time is within class time
                if class_time_obj <= current_time <= class_end_time:
                    print(f"✅ Student {student_id} is in active class: {class_name}")
                    return {
                        'class_name': class_name,
                        'class_date': class_date,
                        'class_time': class_time,
                        'duration': duration
                    }
                else:
                    print(f"❌ Student {student_id} is outside class time")
            except Exception as e:
                print(f"Error parsing time for class {class_name}: {e}")
                continue
        
        print(f"❌ No active class found for student {student_id}")
        return None

    def has_taken_attendance_for_class(self, student_id, class_name, class_date):
        """Check if student has already taken attendance for this class"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM attendance 
            WHERE student_id = ? AND class_name = ? AND date = ?
        ''', (student_id, class_name, class_date))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0

    def predict_spoof_only(self, image):
        """Fast pass: anti-spoof detection only, NO face_recognition (keeps FPS high).
        Merges cached recognition results from the slow recognition_loop thread."""
        try:
            face_detections = self.model_test.get_multiple_bboxes(image, self.max_faces)
            if not face_detections:
                # Clear recognition cache when no faces present
                with self.recognition_lock:
                    self.recognition_cache.clear()
                return []

            results = []

            # Build real-face bbox list to pass to recognition thread
            real_bboxes = []

            for idx, (bbox, detection_conf) in enumerate(face_detections):
                prediction = np.zeros((1, 3))

                for model_name in self.models:
                    model_path = os.path.join(self.model_dir, model_name)
                    h_input, w_input, model_type, scale = parse_model_name(model_name)
                    param = {
                        "org_img": image,
                        "bbox": bbox,
                        "scale": scale,
                        "out_w": w_input,
                        "out_h": h_input,
                        "crop": True,
                    }
                    if scale is None:
                        param["crop"] = False
                    img_crop = self.image_cropper.crop(**param)
                    prediction += self.model_test.predict(img_crop, model_path)

                prediction /= len(self.models)
                is_real = prediction[0][1] > self.threshold
                spoof_confidence = prediction[0][1] if is_real else prediction[0][0]

                # Pull cached recognition result (updated by recognition_loop thread)
                with self.recognition_lock:
                    cached = self.recognition_cache.get(idx, {})
                recognized_name = cached.get('recognizedName')
                recognized_id   = cached.get('recognizedId')
                recognized_conf = cached.get('recognizedConfidence')
                class_info      = cached.get('classInfo')

                if is_real:
                    real_bboxes.append((bbox, idx))

                results.append({
                    'bbox': bbox.tolist() if hasattr(bbox, 'tolist') else bbox,
                    'isReal': bool(is_real),
                    'confidence': float(spoof_confidence),
                    'detectionConfidence': float(detection_conf),
                    'recognizedName': recognized_name,
                    'recognizedId': recognized_id,
                    'recognizedConfidence': recognized_conf,
                    'classInfo': class_info
                })

            # Queue frame + bboxes for the recognition thread (non-blocking)
            self.recognition_frame_queue = image.copy()
            self.recognition_bbox_queue  = real_bboxes

            return results

        except Exception as e:
            print(f"Error in predict_spoof_only: {e}")
            return []

    @staticmethod
    def _face_confidence(face_distance, threshold=0.6):
        """Map a dlib face distance (0=identical) to an intuitive 0..1 confidence.
        A genuine match (~0.3) lands near 0.95; the 0.6 tolerance boundary is ~0.5.
        This is the standard face_recognition curve, not a naive linear scale."""
        if face_distance > threshold:
            linear = (1.0 - face_distance) / ((1.0 - threshold) * 2.0)
            return float(max(0.0, linear))
        linear = 1.0 - (face_distance / (threshold * 2.0))
        return float(linear + (1.0 - linear) * (((linear - 0.5) * 2.0) ** 0.2))

    def recognition_loop(self):
        """Slow background thread: runs face_recognition every `recognition_interval` seconds.
        Completely decoupled from the main detection loop so it never blocks the camera."""
        while self.running:
            time.sleep(self.recognition_interval)

            if not self.detection_active:
                self.real_streak = {}
                continue

            # Grab latest queued data (non-blocking snapshot)
            frame = self.recognition_frame_queue
            bboxes = list(self.recognition_bbox_queue)

            with self.recognition_lock:
                known_encodings = self.known_face_encodings
                known_names = self.known_face_names
                known_ids = self.known_face_ids

            if frame is None or not bboxes or not known_encodings:
                # Nothing live to confirm this cycle → all streaks reset.
                self.real_streak = {}
                continue

            new_cache = {}
            recognized_this_cycle = set()   # student_ids confirmed real+recognized now

            for bbox, face_idx in bboxes:
                x, y, w, h = bbox
                # ── Downscale crop for faster dlib encoding ──────────────────
                face_img = frame[y:y+h, x:x+w]
                scale_factor = 0.5
                small = cv2.resize(face_img,
                                   (max(1, int(w * scale_factor)),
                                    max(1, int(h * scale_factor))))
                face_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                # ────────────────────────────────────────────────────────────

                try:
                    with self.dlib_lock:
                        encodings = face_recognition.face_encodings(face_rgb)
                except Exception:
                    encodings = []

                if not encodings:
                    # Try full-size as fallback
                    try:
                        face_rgb_full = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        with self.dlib_lock:
                            encodings = face_recognition.face_encodings(face_rgb_full)
                    except Exception:
                        encodings = []

                if not encodings:
                    continue

                enc = encodings[0]
                matches = face_recognition.compare_faces(known_encodings, enc)
                distances = face_recognition.face_distance(known_encodings, enc)

                if True in matches:
                    best = int(np.argmin(distances))
                    if distances[best] < 0.55:
                        r_name = known_names[best]
                        r_id   = known_ids[best]
                        # Turn the face-distance into a 0..1 recognition confidence
                        # (distance 0 = identical = 100%, distance 0.55 = cutoff ≈ 0%).
                        rec_conf = self._face_confidence(distances[best])

                        # ── Temporal anti-spoof gate ──────────────────────────
                        # These bboxes only contain faces the liveness model rated
                        # REAL this pass, so reaching here counts as one real cycle.
                        # Count each student at most once per cycle so two boxes of the
                        # same person (e.g. real face + a photo) can't fast-track it.
                        if r_id in recognized_this_cycle:
                            continue
                        recognized_this_cycle.add(r_id)
                        streak = self.real_streak.get(r_id, 0) + 1
                        self.real_streak[r_id] = streak
                        print(f"🔍 Recognition: {r_name} ({r_id}) dist={distances[best]:.3f} "
                              f"conf={rec_conf:.2f} real-streak={streak}/{self.required_real_cycles}")

                        class_info = self.get_student_class_info(r_id)
                        if streak >= self.required_real_cycles:
                            self.take_attendance(r_id, r_name, rec_conf, class_info)
                        # ──────────────────────────────────────────────────────

                        new_cache[face_idx] = {
                            'recognizedName': r_name,
                            'recognizedId': r_id,
                            'recognizedConfidence': rec_conf,
                            'classInfo': class_info
                        }

            # Any student not confirmed real+recognized this cycle loses their streak,
            # so a face must stay continuously live to ever reach the threshold.
            for sid in [s for s in self.real_streak if s not in recognized_this_cycle]:
                del self.real_streak[sid]

            # Atomically replace cache
            with self.recognition_lock:
                self.recognition_cache = new_cache

    def take_attendance(self, student_id, student_name, confidence, class_info):
        """Take attendance for a recognized student"""
        today = date.today().isoformat()
        now = datetime.now().strftime('%H:%M:%S')
        
        # Check if student has a valid class
        if not class_info:
            message = f"Student {student_name} ID {student_id} does not belong to class"
            self.add_attendance_message(message, now, student_name, student_id, None, False)
            return False
        
        # Check if already took attendance for this class today
        if self.has_taken_attendance_for_class(student_id, class_info['class_name'], today):
            print(f"⚠️ Student {student_name} already took attendance for class {class_info['class_name']} today")
            return False
        
        # Check if enough time has passed since last attendance (prevent spam)
        if student_id in self.last_attendance_time:
            last_time = self.last_attendance_time[student_id]
            if (datetime.now() - last_time).seconds < 30:  # 30 seconds cooldown
                print(f"⚠️ Student {student_name} in cooldown period")
                return False
        
        try:
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO attendance (student_id, date, time, status, class_name)
                VALUES (?, ?, ?, ?, ?)
            ''', (student_id, today, now, 'present', class_info['class_name']))
            
            conn.commit()
            conn.close()
            
            # Update tracking
            self.last_attendance_time[student_id] = datetime.now()
            self.detection_stats['attendance_taken'] += 1
            
            # Create attendance message
            message = f"Student {student_name} ID {student_id} did attendance checking successfully for the class: {class_info['class_name']}"
            self.add_attendance_message(message, now, student_name, student_id, class_info['class_name'], True)
            
            print(f"✅ {message}")
            return True
            
        except Exception as e:
            print(f"Error taking attendance for {student_name}: {e}")
            return False

    def add_attendance_message(self, message, timestamp, student_name, student_id, class_name, is_success):
        """Add attendance message to recent messages"""
        self.recent_attendance_messages.append({
            'message': message,
            'timestamp': timestamp,
            'student_name': student_name,
            'student_id': student_id,
            'class_name': class_name,
            'is_success': is_success
        })
        
        # Keep only last 10 messages
        if len(self.recent_attendance_messages) > 10:
            self.recent_attendance_messages.pop(0)

    def detection_loop(self):
        """Main detection loop — runs spoof detection asynchronously on the latest frame.
        Recognition is handled separately by the recognition_loop thread."""
        frame_count = 0
        fps_frame_count = 0
        start_time = time.time()

        while self.running:
            if not self.detection_active or not self.camera_initialized:
                time.sleep(0.05)
                continue

            # Safely grab a copy of the latest camera frame
            with self.camera_lock:
                frame = self.current_frame.copy() if self.current_frame is not None else None

            if frame is None:
                time.sleep(0.01)
                continue

            # Run spoof detection
            faces = self.predict_spoof_only(frame)
            self.current_faces = faces
            self.detection_stats['total_faces']       = len(faces)
            self.detection_stats['real_faces']        = sum(1 for f in faces if f['isReal'])
            self.detection_stats['fake_faces']        = sum(1 for f in faces if not f['isReal'])
            self.detection_stats['recognized_faces']  = sum(1 for f in faces if f['recognizedName'])

            frame_count += 1
            fps_frame_count += 1

            # Recalculate FPS every 30 frames
            if fps_frame_count >= 30:
                elapsed = time.time() - start_time
                self.detection_stats['fps'] = fps_frame_count / elapsed if elapsed > 0 else 0
                fps_frame_count = 0
                start_time = time.time()

            # Slight delay to avoid consuming 100% CPU on a single core
            time.sleep(0.05)

    def start_detection(self):
        """Start the detection loop"""
        if not self.camera_initialized:
            if not self.init_camera():
                return False
        self.detection_active = True
        print("▶️ Detection started")
        return True

    def stop_detection(self):
        """Stop the detection loop and release camera"""
        self.detection_active = False
        self.release_camera()
        print("⏸️ Detection stopped and camera released")

    def capture_photo(self):
        """Capture a photo from the current camera frame"""
        try:
            if self.current_frame is not None:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', self.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                return f'data:image/jpeg;base64,{frame_base64}'
            else:
                return None
        except Exception as e:
            print(f"Error capturing photo: {e}")
            return None

    def setup_api_routes(self):
        """Setup Flask API routes"""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/live')
        def live():
            return render_template('live.html')
        
        @self.app.route('/students')
        def students():
            return render_template('students.html')
        
        @self.app.route('/attendance')
        def attendance():
            return render_template('attendance.html')
        
        @self.app.route('/classes')
        def classes():
            return render_template('classes.html')
        
        @self.app.route('/api/health')
        def health():
            return jsonify({
                'status': 'healthy',
                'system': 'attendance_management',
                'detection_active': self.detection_active,
                'camera_initialized': self.camera_initialized,
                'students_loaded': len(self.known_face_encodings)
            })
        
        @self.app.route('/api/start_detection', methods=['POST'])
        def start_detection_endpoint():
            success = self.start_detection()
            return jsonify({
                'success': success, 
                'message': 'Detection started' if success else 'Failed to start detection'
            })
        
        @self.app.route('/api/stop_detection', methods=['POST'])
        def stop_detection_endpoint():
            self.stop_detection()
            return jsonify({'success': True, 'message': 'Detection stopped'})
        
        @self.app.route('/api/video_feed')
        def video_feed():
            """MJPEG stream — browser sets <img src="/api/video_feed"> once and
            frames flow continuously with no per-frame HTTP overhead."""
            def generate():
                while True:
                    if not self.camera_initialized:
                        time.sleep(0.05)
                        continue
                    
                    with self.camera_lock:
                        frame = self.current_frame.copy() if self.current_frame is not None else None
                        
                    if frame is None:
                        time.sleep(0.05)
                        continue
                        
                    # Encode at moderate quality for bandwidth/speed balance
                    ret, buffer = cv2.imencode(
                        '.jpg', frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 75]
                    )
                    if not ret:
                        time.sleep(0.05)
                        continue
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n'
                        + buffer.tobytes()
                        + b'\r\n'
                    )
                    # Stream cap at ~30 FPS
                    time.sleep(0.033)

            from flask import Response
            return Response(
                generate(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @self.app.route('/api/current_frame')
        def current_frame():
            """Single-frame snapshot (kept for photo capture / fallback)."""
            if self.current_frame is not None:
                _, buffer = cv2.imencode(
                    '.jpg', self.current_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 90]
                )
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                return jsonify({
                    'success': True,
                    'frame': f'data:image/jpeg;base64,{frame_base64}'
                })
            return jsonify({'success': False, 'error': 'No frame available'})
        
        @self.app.route('/api/faces')
        def get_faces():
            return jsonify({
                'success': True,
                'faces': self.current_faces,
                'stats': self.detection_stats,
                'recent_attendance': self.recent_attendance_messages
            })
        
        @self.app.route('/api/capture_photo', methods=['POST'])
        def capture_photo_endpoint():
            try:
                photo_data = self.capture_photo()
                if photo_data:
                    return jsonify({
                        'success': True,
                        'photo': photo_data
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Could not capture photo'
                    })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        
        @self.app.route('/api/students', methods=['GET'])
        def get_students():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT student_id, name, created_at FROM students ORDER BY name')
            students = cursor.fetchall()
            conn.close()
            
            return jsonify({
                'success': True,
                'students': [{'id': s[0], 'name': s[1], 'created_at': s[2]} for s in students]
            })
        
        @self.app.route('/api/students', methods=['POST'])
        def add_student():
            data = request.get_json()
            student_id = data.get('student_id')
            name = data.get('name')
            photo_data = data.get('photo')  # Base64 encoded image
            
            if not all([student_id, name, photo_data]):
                return jsonify({'success': False, 'error': 'Missing required fields'})
            
            try:
                # Decode and process photo
                photo_bytes = base64.b64decode(photo_data.split(',')[1])
                photo_image = Image.open(BytesIO(photo_bytes))
                photo_rgb = cv2.cvtColor(np.array(photo_image), cv2.COLOR_RGB2BGR)
                
                # Get face encodings
                with self.dlib_lock:
                    face_encodings = face_recognition.face_encodings(photo_rgb)
                
                if not face_encodings:
                    return jsonify({'success': False, 'error': 'No face detected in photo'})
                
                face_encoding = face_encodings[0]
                
                # Check for duplicate face
                with self.recognition_lock:
                    known_encodings = self.known_face_encodings
                    known_names = self.known_face_names
                    known_ids = self.known_face_ids

                if known_encodings:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    if True in matches:
                        best_match_index = np.argmin(face_distances)
                        if face_distances[best_match_index] < 0.45:  # Stricter recognition threshold for registration
                            matched_name = known_names[best_match_index]
                            matched_id = known_ids[best_match_index]
                            return jsonify({'success': False, 'error': f'This face is already registered to student: {matched_name} ({matched_id})'})
                
                # Save to database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO students (student_id, name, face_encoding)
                    VALUES (?, ?, ?)
                ''', (student_id, name, pickle.dumps(face_encoding)))
                
                conn.commit()
                conn.close()
                
                # Reload face encodings
                self.load_face_encodings()
                
                return jsonify({'success': True, 'message': 'Student added successfully'})
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/students/<student_id>', methods=['GET'])
        def get_student(student_id):
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT student_id, name, created_at FROM students WHERE student_id = ?', (student_id,))
            student = cursor.fetchone()
            conn.close()
            
            if not student:
                return jsonify({'success': False, 'error': 'Student not found'})
                
            return jsonify({
                'success': True,
                'student': {
                    'id': student[0],
                    'name': student[1],
                    'created_at': student[2]
                }
            })

        @self.app.route('/api/students/<student_id>', methods=['PUT'])
        def update_student(student_id):
            data = request.get_json()
            new_name = data.get('name')
            photo_data = data.get('photo')
            
            if not new_name:
                return jsonify({'success': False, 'error': 'Name is required'})
                
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                if photo_data:
                    # Update with new photo
                    photo_bytes = base64.b64decode(photo_data.split(',')[1])
                    photo_image = Image.open(BytesIO(photo_bytes))
                    photo_rgb = cv2.cvtColor(np.array(photo_image), cv2.COLOR_RGB2BGR)
                    with self.dlib_lock:
                        face_encodings = face_recognition.face_encodings(photo_rgb)
                    if not face_encodings:
                        conn.close()
                        return jsonify({'success': False, 'error': 'No face detected in photo'})
                        
                    face_encoding = face_encodings[0]
                    
                    # Check for duplicate face (excluding current student)
                    with self.recognition_lock:
                        known_encodings = self.known_face_encodings
                        known_names = self.known_face_names
                        known_ids = self.known_face_ids

                    if known_encodings:
                        for i, known_id in enumerate(known_ids):
                            if known_id == student_id:
                                continue # Skip self
                                
                            match = face_recognition.compare_faces([known_encodings[i]], face_encoding, tolerance=0.45)[0]
                            dist = face_recognition.face_distance([known_encodings[i]], face_encoding)[0]
                            if match and dist < 0.45:
                                conn.close()
                                return jsonify({'success': False, 'error': f'This face is already registered to student: {known_names[i]} ({known_id})'})

                    cursor.execute('''
                        UPDATE students SET name = ?, face_encoding = ? WHERE student_id = ?
                    ''', (new_name, pickle.dumps(face_encoding), student_id))
                else:
                    # Update only name
                    cursor.execute('''
                        UPDATE students SET name = ? WHERE student_id = ?
                    ''', (new_name, student_id))
                    
                if cursor.rowcount == 0:
                    conn.close()
                    return jsonify({'success': False, 'error': 'Student not found'})
                    
                conn.commit()
                conn.close()
                
                # Reload face encodings
                self.load_face_encodings()
                
                return jsonify({'success': True, 'message': 'Student updated successfully'})
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/students/<student_id>', methods=['DELETE'])
        def delete_student(student_id):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Check if student exists
                cursor.execute('SELECT name FROM students WHERE student_id = ?', (student_id,))
                student_data = cursor.fetchone()
                
                if not student_data:
                    conn.close()
                    return jsonify({'success': False, 'error': 'Student not found'})
                    
                # Delete student
                cursor.execute('DELETE FROM students WHERE student_id = ?', (student_id,))
                
                # Optional: Delete attendance records for this student
                cursor.execute('DELETE FROM attendance WHERE student_id = ?', (student_id,))
                
                conn.commit()
                conn.close()
                
                # Reload face encodings
                self.load_face_encodings()
                
                return jsonify({'success': True, 'message': f'Student "{student_data[0]}" deleted successfully'})
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/attendance', methods=['GET'])
        def get_attendance():
            date_filter = request.args.get('date', date.today().isoformat())
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT s.name, s.student_id, a.time, a.class_name
                FROM attendance a
                JOIN students s ON a.student_id = s.student_id
                WHERE a.date = ?
                ORDER BY a.time DESC
            ''', (date_filter,))
            
            records = cursor.fetchall()
            conn.close()
            
            return jsonify({
                'success': True,
                'attendance': [{
                    'name': r[0],
                    'student_id': r[1],
                    'time': r[2],
                    'class': r[3] or 'No Class'
                } for r in records]
            })
        
        @self.app.route('/api/classes', methods=['GET'])
        def get_classes():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT id, class_name, student_ids, class_date, class_time, duration_minutes FROM classes')
            classes = cursor.fetchall()
            conn.close()
            
            return jsonify({
                'success': True,
                'classes': [{
                    'id': c[0],
                    'name': c[1],
                    'student_ids': json.loads(c[2]) if c[2] else [],
                    'class_date': c[3],
                    'class_time': c[4],
                    'duration_minutes': c[5]
                } for c in classes]
            })
        
        @self.app.route('/api/classes/<int:class_id>', methods=['GET'])
        def get_class_details(class_id):
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get class info
            cursor.execute('SELECT class_name, student_ids, class_date, class_time, duration_minutes FROM classes WHERE id = ?', (class_id,))
            class_data = cursor.fetchone()
            
            if not class_data:
                conn.close()
                return jsonify({'success': False, 'error': 'Class not found'})
            
            class_name, student_ids_json, class_date, class_time, duration = class_data
            student_ids = json.loads(student_ids_json) if student_ids_json else []
            
            # Get student names
            student_names = []
            if student_ids:
                placeholders = ','.join(['?' for _ in student_ids])
                cursor.execute(f'SELECT name FROM students WHERE student_id IN ({placeholders})', student_ids)
                student_names = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            return jsonify({
                'success': True,
                'class': {
                    'id': class_id,
                    'name': class_name,
                    'student_ids': student_ids,
                    'student_names': student_names,
                    'class_date': class_date,
                    'class_time': class_time,
                    'duration_minutes': duration
                }
            })
        
        @self.app.route("/api/classes/<int:class_id>", methods=["PUT"])
        def update_class(class_id):
            data = request.get_json()
            class_name = data.get("name")
            student_ids = data.get("student_ids", [])
            class_date = data.get("class_date")
            class_time = data.get("class_time")
            duration_minutes = data.get("duration_minutes", 60)
            
            if not all([class_name, class_date, class_time]):
                return jsonify({"success": False, "error": "Class name, date, and time are required"})
            
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Update the class
                cursor.execute("""
                    UPDATE classes 
                    SET class_name = ?, student_ids = ?, class_date = ?, class_time = ?, duration_minutes = ?
                    WHERE id = ?
                """, (class_name, json.dumps(student_ids), class_date, class_time, duration_minutes, class_id))
                
                if cursor.rowcount == 0:
                    conn.close()
                    return jsonify({"success": False, "error": "Class not found"})
                
                conn.commit()
                conn.close()
                
                return jsonify({"success": True, "message": f"Class \"{class_name}\" updated successfully"})
                
            except Exception as e:
                return jsonify({"success": False, "error": str(e)})
        
        @self.app.route("/api/classes/<int:class_id>", methods=["DELETE"])
        def delete_class(class_id):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get class name for confirmation
                cursor.execute("SELECT class_name FROM classes WHERE id = ?", (class_id,))
                class_data = cursor.fetchone()
                
                if not class_data:
                    conn.close()
                    return jsonify({"success": False, "error": "Class not found"})
                
                # Delete the class
                cursor.execute("DELETE FROM classes WHERE id = ?", (class_id,))
                
                conn.commit()
                conn.close()
                
                return jsonify({"success": True, "message": f"Class \"{class_data[0]}\" deleted successfully"})
                
            except Exception as e:
                return jsonify({"success": False, "error": str(e)})
        @self.app.route("/api/classes", methods=["POST"])
        def create_class():
            data = request.get_json()
            class_name = data.get('name')
            student_ids = data.get('student_ids', [])
            class_date = data.get('class_date')
            class_time = data.get('class_time')
            duration_minutes = data.get('duration_minutes', 60)
            
            if not all([class_name, class_date, class_time]):
                return jsonify({'success': False, 'error': 'Class name, date, and time are required'})
            
            if not student_ids:
                return jsonify({'success': False, 'error': 'At least one student must be selected'})
            
            try:
                # Validate that all student IDs exist
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                placeholders = ','.join(['?' for _ in student_ids])
                cursor.execute(f'SELECT student_id FROM students WHERE student_id IN ({placeholders})', student_ids)
                existing_students = [row[0] for row in cursor.fetchall()]
                
                if len(existing_students) != len(student_ids):
                    missing_students = set(student_ids) - set(existing_students)
                    conn.close()
                    return jsonify({'success': False, 'error': f'Students not found: {", ".join(missing_students)}'})
                
                # Create the class
                cursor.execute('''
                    INSERT INTO classes (class_name, student_ids, class_date, class_time, duration_minutes)
                    VALUES (?, ?, ?, ?, ?)
                ''', (class_name, json.dumps(student_ids), class_date, class_time, duration_minutes))
                
                conn.commit()
                conn.close()
                
                return jsonify({'success': True, 'message': f'Class "{class_name}" created successfully with {len(student_ids)} students'})
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

    def run_api_server(self, host='0.0.0.0', port=8000):
        """Run the Flask API server"""
        print(f"🌐 Starting Attendance Management System on http://{host}:{port}")
        print("📡 Available pages:")
        print("  GET  / - Home page")
        print("  GET  /live - Live detection page")
        print("  GET  /students - Manage students page")
        print("  GET  /attendance - Attendance records page")
        print("  GET  /classes - Manage classes page")
        print()
        print("�� Available API endpoints:")
        print("  GET  /api/health - Health check")
        print("  POST /api/start_detection - Start detection")
        print("  POST /api/stop_detection - Stop detection")
        print("  GET  /api/video_feed    - MJPEG live stream (use as <img src>)")
        print("  GET  /api/current_frame - Single JPEG snapshot")
        print("  GET  /api/faces - Get current detection results")
        print("  POST /api/capture_photo - Capture photo from live camera")
        print("  GET  /api/students - Get all students")
        print("  POST /api/students - Add new student")
        print("  GET  /api/attendance - Get attendance records")
        print("  GET  /api/classes - Get all classes")
        print("  GET  /api/classes/<id> - Get class details")
        print("  POST /api/classes - Create new class")
        print()
        print("�� Detection starts OFF - camera is completely free for other uses!")
        
        try:
            self.app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            self.running = False
            self.release_camera()
            print("✅ Shutdown complete")

def main():
    try:
        # Create the attendance management system
        system = AttendanceSystem()
        
        # Run the API server
        system.run_api_server()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
