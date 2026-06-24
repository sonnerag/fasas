#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke test for the anti-spoofing core (no webcam / no Flask)."""
import os, sys, traceback
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

from core.src.anti_spoof_predict import AntiSpoofPredict, Detection
from core.src.generate_patches import CropImage
from core.src.utility import parse_model_name
from config.settings import MODEL_DIR

print("=" * 55)
print("SMOKE TEST: Multi-Face Anti-Spoofing core")
print("=" * 55)

# 1. Load anti-spoof predictor (CPU)
predictor = AntiSpoofPredict(0)
cropper = CropImage()
print("[1] AntiSpoofPredict + CropImage khởi tạo OK, device =", predictor.device)

# 2. Load Caffe face detector
print("[2] Face detector (RetinaFace Caffe) đã load:",
      predictor.detector is not None)

# 3. List models
models = [m for m in os.listdir(MODEL_DIR) if m.endswith(".pth")]
print("[3] Tìm thấy", len(models), "model:", models)

# 4. Run each model on a synthetic 80x80 face crop
dummy = (np.random.rand(80, 80, 3) * 255).astype("uint8")
prediction = np.zeros((1, 3))
for m in models:
    h, w, mtype, scale = parse_model_name(m)
    out = predictor.predict(dummy, os.path.join(MODEL_DIR, m))
    print(f"    -> {m}: type={mtype} input={w}x{h} softmax={np.round(out[0],4)}")
    prediction += out
prediction /= len(models)
label = int(np.argmax(prediction))
print("[4] Dự đoán gộp:", np.round(prediction[0], 4),
      "=> nhãn", label, "(0=2D-fake, 1=real, 2=3D-fake)")

# 5. Run the detector on a blank frame (should simply return no faces, not crash)
blank = np.zeros((480, 640, 3), dtype="uint8")
faces = predictor.get_multiple_bboxes(blank, max_faces=5)
print("[5] get_multiple_bboxes trên frame trống trả về", len(faces), "khuôn mặt (đúng = 0)")

print("=" * 55)
print("✅ CORE PIPELINE CHẠY THÀNH CÔNG")
print("=" * 55)
