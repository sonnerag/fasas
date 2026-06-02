# Graduation Report: Multi-Face Anti-Spoofing & Attendance Management System

## Part 2: Core Models and Libraries

### 1. Face Detection
The system employs a robust approach for face detection to ensure accuracy before further processing.
*   **Model/Architecture**: The primary detector is based on the **RetinaFace** architecture (specifically `Widerface-RetinaFace.caffemodel`), utilizing an SSD (Single Shot MultiBox Detector) framework. 
*   **Library Integration**: We utilize OpenCV's Deep Neural Network (`cv2.dnn.readNetFromCaffe`) module to load the pre-trained Caffe models. This provides a highly optimized, CPU/GPU-friendly inference mechanism that rapidly identifies multiple bounding boxes and confidence scores for faces within the frame.
*   **Fallback / Alignment**: For identity matching, the system also leverages the `dlib` library (via the `face_recognition` package), which uses Histogram of Oriented Gradients (HOG) combined with a Linear SVM, or alternatively a CNN-based detector, to robustly localize facial landmarks and crop the face accurately.

### 2. Face Anti-Spoofing (Liveness Detection)

Face anti-spoofing is an essential security component of the proposed attendance management system. Its purpose is to verify whether a detected face belongs to a real live person rather than a spoofing attack such as a printed photograph, replayed video, mobile phone display, or mask. Traditional face recognition systems are vulnerable to presentation attacks if liveness verification is not applied before identity recognition.

To address this problem, the proposed system integrates the Silent Face Anti-Spoofing framework, which provides a lightweight and efficient deep learning solution for real-time liveness detection. The framework utilizes MiniFASNet-based convolutional neural network architectures optimized for practical anti-spoofing applications.

#### Silent Face Anti-Spoofing Framework

The system uses the Silent Face Anti-Spoofing framework as the core anti-spoofing solution. This framework is implemented using PyTorch and is specifically designed for lightweight and real-time face liveness verification.

The framework includes:

* Pre-trained anti-spoofing models
* Multi-scale inference mechanisms
* Face patch generation
* Prediction aggregation
* Image preprocessing utilities

The framework was selected because it provides:

* High inference speed
* Low computational cost
* Real-time webcam compatibility
* CPU-friendly execution
* Strong resistance against common presentation attacks

The framework is integrated into the attendance system after the face detection stage and before the face recognition stage.

#### MiniFASNet Architecture

The core neural network architecture used inside the Silent Face Anti-Spoofing framework is MiniFASNet.

MiniFASNet is a lightweight Convolutional Neural Network (CNN) designed specifically for face anti-spoofing tasks. The model contains approximately 0.43 million parameters, allowing the system to perform real-time inference efficiently without requiring dedicated GPU hardware.

Compared with larger deep learning models, MiniFASNet achieves a balance between:

* Detection accuracy
* Processing speed
* Memory efficiency
* Real-time performance

This makes the architecture suitable for practical attendance management systems operating on standard computers.

#### Depth-Wise Separable Convolutions

MiniFASNet uses depth-wise separable convolutions instead of standard convolution layers.

Traditional convolution operations process all image channels simultaneously, resulting in high computational complexity. In contrast, depth-wise separable convolutions divide the process into:

* Depth-wise convolution
* Point-wise convolution

This design significantly reduces:

* Floating point operations (FLOPs)
* Memory consumption
* Inference time

while preserving important spatial and texture features required for spoof detection.

As a result, the anti-spoofing system can operate efficiently in real-time webcam environments.

#### Residual Learning

The network architecture also incorporates residual connections inspired by ResNet architectures.

Residual blocks allow feature information to bypass intermediate layers and merge directly with later outputs:

```python
short_cut + x
```

This mechanism improves gradient flow during training and helps the network learn deeper feature representations more effectively.

Residual learning contributes to:

* Better training stability
* Improved feature extraction
* Higher generalization capability

especially when distinguishing subtle differences between real faces and spoof attacks.

#### Squeeze-and-Excitation (SE) Modules

The implemented MiniFASNet variants include Squeeze-and-Excitation (SE) modules such as `MiniFASNetV2SE`.

The SE module dynamically learns which feature channels are most important for spoof detection. It performs:

1. Global feature compression using adaptive average pooling
2. Channel importance estimation
3. Feature channel recalibration

This mechanism enables the network to focus more strongly on discriminative spoofing artifacts such as:

* Screen reflections
* Printing textures
* Moiré patterns
* Edge inconsistencies
* Illumination artifacts

The SE module improves the network’s ability to separate genuine faces from presentation attacks.

#### Face Cropping and Patch Generation

The anti-spoofing pipeline uses the `CropImage` class implemented in `generate_patches.py` to generate image patches for analysis.

Instead of using only the exact face bounding box generated by RetinaFace, the system expands the facial region using scale multipliers:

```python
box_w * scale
```

This expanded cropping strategy is important because spoofing clues often exist outside the central facial area.

Examples include:

* Borders of printed photographs
* Mobile phone edges
* Reflection artifacts
* Background inconsistencies
* Hand-held spoofing devices

By including surrounding contextual information, the anti-spoofing models achieve improved detection performance.

#### Boundary Protection

The cropping implementation also contains boundary correction logic to ensure valid image coordinates during real-time processing.

The system automatically adjusts crop coordinates if the expanded region exceeds image boundaries. This prevents:

* Array out-of-bounds errors
* Invalid crops
* Runtime crashes

during webcam inference.

#### Multi-Scale Ensemble Strategy

One of the most important design features of the proposed system is the use of a multi-scale ensemble strategy.

The system loads multiple anti-spoofing models trained using different crop scales and input resolutions. For a single detected face, several image patches are generated:

* Tight facial crops
* Medium-scale crops
* Wide contextual crops

Each image patch is evaluated independently by its corresponding MiniFASNet model.

The prediction scores are then combined through averaging:

```python
prediction /= len(self.models)
```

This ensemble strategy improves robustness because different spoofing artifacts become visible at different spatial scales.

For example:

* Tight crops emphasize skin texture quality
* Wider crops reveal phone borders or paper edges

Combining predictions from multiple scales improves reliability and reduces false positives.

#### Integration with Attendance System

The anti-spoofing module is positioned between face detection and face recognition within the attendance pipeline:

```text
Face Detection
→ Face Cropping
→ Anti-Spoofing Verification
→ Face Recognition
→ Attendance Logging
```

Only faces classified as real are forwarded to the recognition stage. Faces identified as spoof attacks are immediately rejected and prevented from accessing the attendance system.

This design significantly improves system security and prevents unauthorized attendance registration using fake facial inputs.

### 3. Face Recognition
Once a face is verified as "Real," the system determines the individual's identity.
*   **Model/Architecture**: Identity verification is handled by the **`face_recognition`** library, which wraps around `dlib`'s state-of-the-art face recognition model. This model is a ResNet network trained on millions of images.
*   **Implementation**: It maps the cropped facial image into a 128-dimensional latent space (face encoding). The system computes the Euclidean distance between the live encoding and the pre-computed encodings stored in our database. A threshold (e.g., `< 0.6` for registration, `< 0.8` for attendance) is used to confirm the identity.

---

## Part 3: Architecture, Pipeline, Key Features, and Implementation Details

### System Architecture
The application is built on a hybrid architecture consisting of a Python/Flask backend and an HTML/JS/CSS frontend.
*   **Backend Server**: A Flask REST API handles business logic, video streaming, database operations (SQLite), and the orchestration of the AI models.
*   **Database Schema**: SQLite manages three primary entities: `students` (storing 128-d encodings and metadata), `classes` (schedules and enrollment), and `attendance` (logging timestamps and status).

### The Execution Pipeline
1.  **Frame Acquisition**: The system continuously captures frames from the webcam.
2.  **Detection Phase**: The frame is downscaled, and the SSD detector localizes multiple face bounding boxes.
3.  **Liveness Verification**: Each detected face is cropped and passed through the PyTorch MiniFASNet ensemble to evaluate its liveness score.
4.  **Identity Matching (Asynchronous)**: Real faces are queued for the face recognition module, which extracts the 128-d embedding and queries the SQLite database for matches.
5.  **Attendance Logging**: If matched and within the scheduled class time, the backend records the attendance and updates the web interface via API polling.

### Key Features
*   **Multi-Face Processing**: Capable of detecting and tracking up to 5 faces simultaneously in real-time.
*   **Real-time Video Streaming**: Implements MJPEG streaming over HTTP, delivering a smooth, low-latency video feed directly to the web browser.
*   **Dynamic Class Scheduling**: Validates attendance against real-time class schedules, preventing false check-ins outside of designated hours.

### Implementation Details: Asynchronous Processing
To maintain high application responsiveness, the backend is highly multi-threaded:
*   `CameraReaderLoop`: Continuously flushes the OpenCV buffer to guarantee zero-delay frames.
*   `DetectionLoop`: Runs fast spoof-detection calculations on the latest frame to ensure smooth bounding box rendering on the UI.
*   `RecognitionLoop`: A decoupled thread that periodically (e.g., every 2 seconds) processes face embeddings. This prevents the heavy `dlib` recognition process from dropping the main camera frame rate.

---

## Part 4: Technical Challenges and Solutions

### Challenge 1: Real-Time Performance Bottlenecks
*   **Issue**: Running Face Detection, Anti-Spoofing, and Face Recognition sequentially on a single frame caused severe latency, dropping the FPS to unplayable levels (often < 5 FPS).
*   **Solution**: We implemented a decoupled, multi-threaded architecture. By separating the fast Anti-Spoofing pass (UI bounding boxes) from the slow Face Recognition pass (identity matching), the system remains fluid. The recognition thread runs in the background and updates a shared cache atomically (`self.recognition_cache`), merging identity data with the live video feed seamlessly.

### Challenge 2: Video Buffer Lag
*   **Issue**: OpenCV's `VideoCapture` buffer would queue old frames if the processing loop took too long, resulting in a noticeable delay (lag) between real-world movement and the video feed.
*   **Solution**: A dedicated background `camera_loop` thread was introduced. This thread's sole purpose is to continuously call `cap.read()` as fast as possible, ensuring that whenever the processing thread requests a frame, it receives the absolute most recent physical frame.

### Challenge 3: Environment and Dependency Conflicts
*   **Issue**: Integrating legacy models (Caffe/PyTorch) with modern web frameworks (Flask) and strict C++ bindings (dlib) led to severe dependency clashes, particularly around `protobuf`, `torch`, and `numpy` versions.
*   **Solution**: We established a highly controlled virtual environment (`venv`) with a strict `requirements.txt`. We aligned PyTorch (`>=1.2.0`), OpenCV (`>=4.2.0`), and specifically handled `face_recognition` constraints to guarantee stability across different hardware setups.

---

## Part 5: Conclusion

The development of the Multi-Face Anti-Spoofing and Attendance Management System successfully bridges the gap between advanced deep learning concepts and practical software engineering. By integrating **RetinaFace** for detection, **MiniFASNet** for presentation attack detection, and **dlib-based ResNet** for identity verification, the project delivers a highly secure, automated attendance solution.

The primary engineering success lies in overcoming real-time processing constraints through asynchronous threading and optimized pipeline design, resulting in a smooth, responsive web application. Future improvements could focus on migrating the database to a scalable cloud solution (like PostgreSQL/Firebase), integrating GPU acceleration via TensorRT for edge devices, and expanding the frontend to a fully-fledged PWA (Progressive Web App) for mobile administrative access. 

This project demonstrates a comprehensive understanding of computer vision, backend API design, and system optimization, fulfilling the core objectives of the graduation requirements.
