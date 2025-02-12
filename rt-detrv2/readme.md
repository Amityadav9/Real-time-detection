# Real-Time Object Detection with YOLO and RT-DETRv2

This repository contains implementations of real-time object detection using two popular models: YOLO and RT-DETRv2. The code is designed to process video files and perform real-time object detection.

## Overview

- **YOLO**: A widely-used real-time object detection system known for its speed and accuracy.
- **RT-DETRv2**: A state-of-the-art real-time object detection model that builds upon the original RT-DETR architecture, introducing selective multi-scale feature extraction and improved training strategies for better performance while maintaining real-time capabilities.

## Features

- Real-time object detection with bounding boxes
- Multi-class object recognition
- FPS (Frames Per Second) monitoring
- GPU acceleration support
- Customizable confidence thresholds
- Color-coded visualization
- Optional image captioning (in combined version with RT-DETRv2)

## Implementation Details

### YOLO Implementation

1. **Basic Object Detection (`yolo_video_detection.py`)**
   - Processes video files using the YOLO model.
   - Saves annotated video with detections.

   ```bash
   python yolo_video_detection.py
Real-Time Object Detection (yolo_live_detection.py)

Performs real-time object detection using a webcam.
Displays annotated video in real-time.

python yolo_live_detection.py
RT-DETRv2 Implementation
Basic Object Detection (rt_detrv2_video_detection.py)

Processes video files using the RT-DETRv2 model.
Saves annotated video with detections.

python rt_detrv2_video_detection.py
Real-Time Object Detection (rt_detrv2_live_detection.py)

Performs real-time object detection using a webcam.
Displays annotated video in real-time.

python rt_detrv2_live_detection.py
Combined Detection and Captioning (detection_caption.py)

Combines object detection with image captioning using the BLIP model.
Provides scene descriptions along with detections.

python detection_caption.py
Differences Between YOLO and RT-DETRv2
Ease of Use: YOLO is generally easier to set up and use, especially for real-time applications, due to its straightforward API and extensive documentation.

Performance: YOLO models are known for their speed and accuracy in real-time object detection tasks. RT-DETRv2 might offer different trade-offs in terms of accuracy and speed, depending on the specific model and use case.

Model Size and Requirements: YOLO models are typically lighter and can run efficiently on various hardware, including edge devices. RT-DETRv2 models might be more resource-intensive.

Customization: Both frameworks allow for customization, but YOLO might be more flexible for quick adjustments and tuning.

Installation
Prerequisites
Python 3.8 or higher
CUDA-compatible GPU (recommended)
Webcam or video input device
Required Packages

pip install torch torchvision
pip install transformers
pip install opencv-python
pip install pillow
pip install numpy
pip install ultralytics

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
