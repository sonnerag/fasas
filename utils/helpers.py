#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for Multi-Face Anti-Spoofing System
"""

import cv2
import numpy as np
import time
import os
from typing import List, Tuple, Optional

def validate_model_paths(model_dir: str, detection_model_dir: str) -> bool:
    """Validate that required model files exist"""
    if not os.path.exists(model_dir):
        print(f"Error: Anti-spoofing models directory not found: {model_dir}")
        return False
    
    if not os.path.exists(detection_model_dir):
        print(f"Error: Detection models directory not found: {detection_model_dir}")
        return False
    
    # Check for required detection model files
    required_files = [
        "Widerface-RetinaFace.caffemodel",
        "deploy.prototxt"
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(detection_model_dir, file)):
            print(f"Error: Required detection model file not found: {file}")
            return False
    
    return True

def calculate_fps(start_time: float, frame_count: int) -> float:
    """Calculate current FPS"""
    if frame_count == 0:
        return 0.0
    elapsed_time = time.time() - start_time
    return frame_count / elapsed_time if elapsed_time > 0 else 0.0

def draw_info_panel(frame: np.ndarray, info: dict, show_info: bool = True) -> np.ndarray:
    """Draw information panel on frame"""
    if not show_info:
        return frame
    
    height, width = frame.shape[:2]
    panel_height = 120
    panel_width = 300
    
    # Create semi-transparent panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw text information
    y_offset = 30
    line_height = 20
    
    info_texts = [
        f"FPS: {info.get('fps', 0):.1f}",
        f"Faces: {info.get('face_count', 0)}/{info.get('max_faces', 5)}",
        f"Threshold: {info.get('threshold', 0.5):.2f}",
        f"Status: {info.get('status', 'Running')}"
    ]
    
    for i, text in enumerate(info_texts):
        cv2.putText(frame, text, (20, y_offset + i * line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def draw_face_info(frame: np.ndarray, face_info: dict, face_idx: int) -> np.ndarray:
    """Draw information for a single detected face"""
    bbox = face_info['bbox']
    is_real = face_info['is_real']
    spoof_conf = face_info['spoof_confidence']
    det_conf = face_info['detection_confidence']
    
    # Choose color based on real/fake
    color = (0, 255, 0) if is_real else (0, 0, 255)  # Green for real, Red for fake
    
    # Draw bounding box
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Prepare text
    label = "REAL" if is_real else "FAKE"
    face_text = f"Face {face_idx + 1}"
    spoof_text = f"{label}: {spoof_conf:.3f}"
    det_text = f"Det: {det_conf:.3f}"
    
    # Draw text background
    text_y = y - 10 if y > 30 else y + h + 20
    cv2.rectangle(frame, (x, text_y - 20), (x + 200, text_y + 40), (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, face_text, (x + 5, text_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, spoof_text, (x + 5, text_y + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.putText(frame, det_text, (x + 5, text_y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame

def get_available_cameras() -> List[int]:
    """Get list of available camera indices"""
    available_cameras = []
    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def print_system_info():
    """Print system information and available resources"""
    print("=" * 50)
    print("Multi-Face Anti-Spoofing System")
    print("=" * 50)
    
    # Check available cameras
    cameras = get_available_cameras()
    print(f"Available cameras: {cameras}")
    
    # Check OpenCV version
    print(f"OpenCV version: {cv2.__version__}")
    
    # Check if CUDA is available (if PyTorch is installed)
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
    except ImportError:
        print("PyTorch not available")
    
    print("=" * 50)

def create_output_directory(output_dir: str = "output") -> str:
    """Create output directory if it doesn't exist"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir
