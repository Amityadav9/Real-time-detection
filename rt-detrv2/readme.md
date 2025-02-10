# Real-Time Object Detection with RT-DETRv2

This repository contains two implementations of real-time object detection using the RT-DETRv2 model. The first implementation focuses solely on object detection, while the second combines object detection with image captioning using the BLIP model.

## Overview

RT-DETRv2 is a state-of-the-art real-time object detection model that builds upon the original RT-DETR architecture. It introduces selective multi-scale feature extraction and improved training strategies for better performance while maintaining real-time capabilities.

### Features

- Real-time object detection with bounding boxes
- Multi-class object recognition
- FPS (Frames Per Second) monitoring
- GPU acceleration support
- Customizable confidence thresholds
- Color-coded visualization
- Optional image captioning (in combined version)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Webcam or video input device

### Required Packages

```bash
pip install torch torchvision
pip install transformers
pip install opencv-python
pip install pillow
pip install numpy
```

## Implementation Details

### Version 1: Basic Object Detection (`RealTimeObjectDetector`)

This implementation focuses solely on real-time object detection. The main components include:

1. Model Initialization:
```python
detector = RealTimeObjectDetector(
    model_name="PekingU/rtdetr_v2_r50vd",
    confidence_threshold=0.5
)
```

2. Key Features:
- Real-time object detection
- Confidence-based filtering
- Automatic color generation for different object classes
- FPS monitoring
- Interactive camera feed

### Version 2: Combined Detection and Captioning (`RealTimeDetectionAndCaption`)

This implementation combines object detection with image captioning. The main components include:

1. Model Initialization:
```python
detector = RealTimeDetectionAndCaption(
    detection_model="PekingU/rtdetr_v2_r50vd",
    caption_model="Salesforce/blip-image-captioning-large",
    confidence_threshold=0.5
)
```

2. Additional Features:
- Scene description generation
- Dual model processing (detection + captioning)
- Periodic caption updates
- Enhanced visualization with both detections and descriptions

## Usage

### Basic Object Detection

```python
from object_detection import RealTimeObjectDetector

# Create detector instance
detector = RealTimeObjectDetector()

# Start real-time detection
detector.start_detection(camera_index=0, display_fps=True)
```

### Combined Detection and Captioning

```python
from detection_caption import RealTimeDetectionAndCaption

# Create detector instance
detector = RealTimeDetectionAndCaption()

# Start real-time detection and captioning
detector.start_realtime_detection(camera_index=0, display_fps=True)
```

## Performance Considerations

### Hardware Requirements

- Minimum: 8GB RAM, CUDA-compatible GPU with 6GB VRAM
- Recommended: 16GB RAM, CUDA-compatible GPU with 8GB+ VRAM

### Performance Tips

1. GPU Memory Management:
   - Monitor GPU memory usage
   - Adjust batch sizes if needed
   - Consider using lower resolution for smoother performance

2. Processing Speed:
   - Adjust confidence threshold to balance accuracy and speed
   - Modify caption generation interval in combined version
   - Use appropriate frame resolution

3. Optimization Options:
   - Reduce frame resolution for faster processing
   - Adjust the caption_interval in combined version
   - Use FP16 (half-precision) if supported by your GPU

## Troubleshooting

### Common Issues and Solutions

1. Memory Errors:
   - Reduce frame resolution
   - Increase caption generation interval
   - Close other GPU-intensive applications

2. Low FPS:
   - Lower the confidence threshold
   - Reduce input resolution
   - Use a more powerful GPU

3. Camera Issues:
   - Check camera index
   - Verify camera permissions
   - Test with different resolution settings

## Customization

### Adjustable Parameters

1. Detection Settings:
```python
detector = RealTimeObjectDetector(
    model_name="PekingU/rtdetr_v2_r50vd",
    confidence_threshold=0.5  # Adjust for detection sensitivity
)
```

2. Camera Settings:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Adjust resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

3. Caption Settings (Combined Version):
```python
self.caption_interval = 30  # Adjust caption update frequency
```

## License and Attribution

This implementation uses the following models:
- RT-DETRv2: Created by PekingU
- BLIP (in combined version): Created by Salesforce

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## References

- RT-DETRv2 Paper: [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer]
- BLIP Model: [Salesforce BLIP Image Captioning]
- HuggingFace Transformers Documentation
