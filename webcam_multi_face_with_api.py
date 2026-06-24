#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Multi-Face Anti-Spoofing with Web API
Runs the powerful Tkinter detection and serves results via HTTP API
"""

import cv2
import numpy as np
import time
import os
import sys
import json
import threading
from flask import Flask, jsonify, request
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.src.anti_spoof_predict import AntiSpoofPredict
from core.src.generate_patches import CropImage
from core.src.utility import parse_model_name
from config.settings import *

class MultiFaceWebcamWithAPI:
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
        
        # Camera setup
        self.cap = cv2.VideoCapture(DEFAULT_CAMERA_ID)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {DEFAULT_CAMERA_ID}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Camera {DEFAULT_CAMERA_ID} initialized successfully")
        
        # Detection state
        self.current_frame = None
        self.current_faces = []
        self.is_detecting = False
        self.detection_stats = {
            'total_faces': 0,
            'real_faces': 0,
            'fake_faces': 0,
            'fps': 0
        }
        
        # Threading
        self.detection_thread = None
        self.running = False
        
        # Flask API
        self.app = Flask(__name__)
        CORS(self.app)
        self.setup_api_routes()
    
    def setup_api_routes(self):
        """Setup Flask API routes"""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'models_loaded': len(self.models),
                'models': self.models,
                'camera_active': self.cap.isOpened(),
                'detection_active': self.is_detecting
            })
        
        @self.app.route('/api/current_frame', methods=['GET'])
        def get_current_frame():
            if self.current_frame is not None:
                # Convert frame to base64
                _, buffer = cv2.imencode('.jpg', self.current_frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                return jsonify({
                    'success': True,
                    'frame': f'data:image/jpeg;base64,{frame_base64}',
                    'timestamp': time.time()
                })
            return jsonify({'success': False, 'error': 'No frame available'})
        
        @self.app.route('/api/faces', methods=['GET'])
        def get_faces():
            return jsonify({
                'success': True,
                'faces': self.current_faces,
                'stats': self.detection_stats,
                'timestamp': time.time()
            })
        
        @self.app.route('/api/start_detection', methods=['POST'])
        def start_detection():
            if not self.is_detecting:
                self.start_detection()
                return jsonify({'success': True, 'message': 'Detection started'})
            return jsonify({'success': False, 'message': 'Detection already running'})
        
        @self.app.route('/api/stop_detection', methods=['POST'])
        def stop_detection():
            if self.is_detecting:
                self.stop_detection()
                return jsonify({'success': True, 'message': 'Detection stopped'})
            return jsonify({'success': False, 'message': 'Detection not running'})
        
        @self.app.route('/api/settings', methods=['GET'])
        def get_settings():
            return jsonify({
                'maxFaces': self.max_faces,
                'threshold': self.threshold,
                'deviceId': self.device_id
            })
        
        @self.app.route('/api/settings', methods=['POST'])

        @self.app.route("/web_app_simple.html")
        def web_interface():
            return self.app.send_static_file("web_app_simple.html")

        @self.app.route("/")
        def index():
            return self.app.send_static_file("web_app_simple.html")
        def update_settings():
            data = request.get_json()
            if 'maxFaces' in data:
                self.max_faces = max(1, min(10, int(data['maxFaces'])))
            if 'threshold' in data:
                self.threshold = max(0.1, min(1.0, float(data['threshold'])))
            return jsonify({'success': True, 'settings': {
                'maxFaces': self.max_faces,
                'threshold': self.threshold
            }})
    
    def predict_faces(self, image):
        """Predict if faces are real or fake for multiple faces"""
        try:
            # Get multiple face bounding boxes
            face_detections = self.model_test.get_multiple_bboxes(image, self.max_faces)
            
            if not face_detections:
                return []
            
            results = []
            
            for bbox, detection_conf in face_detections:
                prediction = np.zeros((1, 3))
                
                # Process through each model
                for model_name in self.models:
                    model_path = os.path.join(self.model_dir, model_name)
                    
                    # Crop image for this model
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
                
                # Average predictions
                prediction /= len(self.models)
                
                # Determine if face is real or fake
                is_real = prediction[0][1] > self.threshold
                spoof_confidence = prediction[0][1] if is_real else prediction[0][0]
                
                results.append({
                    'bbox': bbox.tolist() if hasattr(bbox, 'tolist') else bbox,
                    'isReal': bool(is_real),
                    'confidence': float(spoof_confidence),
                    'detectionConfidence': float(detection_conf)
                })
            
            return results
            
        except Exception as e:
            print(f"Error in predict_faces: {e}")
            return []
    
    def detection_loop(self):
        """Main detection loop running in separate thread"""
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break
            
            # Store current frame
            self.current_frame = frame.copy()
            
            # Run detection every few frames for performance
            if frame_count % 2 == 0:  # Process every 2nd frame
                faces = self.predict_faces(frame)
                self.current_faces = faces
                
                # Update stats
                self.detection_stats['total_faces'] = len(faces)
                self.detection_stats['real_faces'] = len([f for f in faces if f['isReal']])
                self.detection_stats['fake_faces'] = len([f for f in faces if not f['isReal']])
            
            frame_count += 1
            
            # Calculate FPS
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                self.detection_stats['fps'] = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
            
            time.sleep(0.033)  # ~30 FPS
    
    def start_detection(self):
        """Start the detection loop"""
        if not self.is_detecting:
            self.is_detecting = True
            self.running = True
            self.detection_thread = threading.Thread(target=self.detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            print("Detection started")
    
    def stop_detection(self):
        """Stop the detection loop"""
        if self.is_detecting:
            self.running = False
            self.is_detecting = False
            if self.detection_thread:
                self.detection_thread.join()
            print("Detection stopped")
    
    def run_api_server(self, host='0.0.0.0', port=8000):
        """Run the Flask API server"""
        print(f"🌐 Starting API server on http://{host}:{port}")
        print("📡 Available endpoints:")
        print("  GET  /api/health - Health check")
        print("  GET  /api/current_frame - Get current camera frame")
        print("  GET  /api/faces - Get current detection results")
        print("  POST /api/start_detection - Start detection")
        print("  POST /api/stop_detection - Stop detection")
        print("  GET  /api/settings - Get current settings")
        print("  POST /api/settings - Update settings")
        print()
        print("🎯 Auto-starting detection...")
        self.start_detection()
        
        try:
            self.app.run(host=host, port=port, debug=False, threaded=True)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            self.stop_detection()
            self.cap.release()
            print("✅ Shutdown complete")

def main():
    try:
        # Create the multi-face anti-spoofing system with API
        system = MultiFaceWebcamWithAPI()
        
        # Run the API server
        system.run_api_server()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
