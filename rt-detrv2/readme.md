# RT-DETRv2 Object Detection

## Overview
This module provides real-time object detection using the RT-DETRv2 (Real-Time Detection Transformer v2) model. RT-DETRv2 is a state-of-the-art object detection model known for its efficiency and accuracy.

## Features
- Real-time object detection
- Live webcam detection
- Video file processing
- GPU acceleration support
- Customizable confidence threshold

## Installation Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Webcam or video input device

## Required Packages
```bash
pip install torch torchvision
pip install transformers
pip install opencv-python
pip install pillow
pip install numpy
```

## Usage

### Live Detection
Run real-time detection using your webcam:
```bash
python rt_detrv2_live_detection.py
```

### Video File Detection
Process a video file and save annotated output:
```bash
python rt_detrv2_video_detection.py
```

## Customization
You can modify detection parameters in the script:
```python
detector = RealTimeObjectDetector(
    model_name="PekingU/rtdetr_v2_r50vd",  # Change model
    confidence_threshold=0.5  # Adjust detection sensitivity
)
```

## Performance Considerations
- Minimum: 8GB RAM, CUDA-compatible GPU with 6GB VRAM
- Recommended: 16GB RAM, CUDA-compatible GPU with 8GB+ VRAM

### Optimization Tips
- Adjust confidence threshold
- Reduce frame resolution
- Use GPU acceleration

## Troubleshooting
- Ensure all dependencies are installed
- Check GPU compatibility
- Verify camera permissions

## References
- [RT-DETRv2 Paper](https://arxiv.org/abs/your-paper-link)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
