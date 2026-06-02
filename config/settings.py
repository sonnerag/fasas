# Configuration settings for Multi-Face Anti-Spoofing System

# Model paths
MODEL_DIR = "./models/resources/anti_spoof_models"
DETECTION_MODEL_DIR = "./models/resources/detection_model"

# Detection settings
DEFAULT_MAX_FACES = 5
DEFAULT_DETECTION_CONFIDENCE = 0.6
DEFAULT_SPOOF_THRESHOLD = 0.5

# Camera settings
DEFAULT_CAMERA_ID = 0
DEFAULT_DEVICE_ID = 0  # 0 for CPU, >0 for GPU

# Display settings
SHOW_FPS = True
SHOW_FACE_COUNT = True
SHOW_THRESHOLD = True
SHOW_MAX_FACES = True

# Colors (BGR format for OpenCV)
COLOR_REAL = (0, 255, 0)      # Green
COLOR_FAKE = (0, 0, 255)      # Red
COLOR_TEXT = (255, 255, 255)  # White
COLOR_INFO = (0, 255, 255)    # Yellow

# Font settings
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Performance settings
TARGET_FPS = 30
FRAME_SKIP = 1  # Process every Nth frame for better performance
