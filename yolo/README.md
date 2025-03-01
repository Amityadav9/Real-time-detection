# YOLO Object Detection

## Overview
This module provides real-time object detection using YOLO (You Only Look Once), a popular and efficient object detection system known for its speed and accuracy.

## Features
- Real-time object detection
- Live webcam detection
- Video file processing
- Multiple pre-trained models
- GPU acceleration support

## Installation Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Webcam or video input device

## Required Packages
```bash
pip install ultralytics
pip install opencv-python
```

## Usage

### Live Detection
Run real-time detection using your webcam:
```bash
python yolo_live_detection.py
```

### Video File Detection
Process a video file and save annotated output:
```bash
python yolo_video_detection.py
```

## Model Selection
YOLO offers various pre-trained models. In the current implementation:
```python
model = YOLO("yolo11n.pt")  # Lighweight YOLO model
```

You can easily switch models by changing the model path:
- `yolo11n.pt`: Nano version (lightest, fastest)
- `yolov8s.pt`: Small version
- `yolov8m.pt`: Medium version
- `yolov8l.pt`: Large version
- `yolov8x.pt`: Extra large version

## Performance Considerations
- Minimum: 4GB RAM, integrated GPU
- Recommended: 8GB RAM, dedicated GPU

### Optimization Tips
- Choose an appropriate model size
- Adjust confidence threshold
- Reduce frame resolution
- Use GPU acceleration

## Troubleshooting
- Ensure Ultralytics package is installed
- Check camera permissions
- Verify GPU drivers

## References
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [YOLO Paper](https://arxiv.org/abs/your-paper-link)

