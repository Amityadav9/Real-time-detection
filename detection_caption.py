import cv2
import torch
from transformers import (
    RTDetrV2ForObjectDetection,
    RTDetrImageProcessor,
    BlipProcessor,
    BlipForConditionalGeneration,
)
import numpy as np
from PIL import Image
import time


class RealTimeDetectionAndCaption:
    def __init__(
        self,
        detection_model="PekingU/rtdetr_v2_r50vd",
        caption_model="Salesforce/blip-image-captioning-large",
        confidence_threshold=0.5,
    ):
        """
        Initialize both detection and captioning models
        """
        # Initialize object detection
        self.det_processor = RTDetrImageProcessor.from_pretrained(detection_model)
        self.det_model = RTDetrV2ForObjectDetection.from_pretrained(detection_model)

        # Initialize image captioning
        self.cap_processor = BlipProcessor.from_pretrained(caption_model)
        self.cap_model = BlipForConditionalGeneration.from_pretrained(caption_model)

        # Move models to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.det_model = self.det_model.to(self.device)
        self.cap_model = self.cap_model.to(self.device)

        self.confidence_threshold = confidence_threshold
        self.colors = self._generate_colors(len(self.det_model.config.id2label))

        # Caption generation settings
        self.caption_interval = 30  # Generate caption every 30 frames
        self.current_caption = "Analyzing scene..."

    def _generate_colors(self, num_classes):
        """Generate distinct colors for visualization"""
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            saturation = 0.8
            value = 0.9

            c = value * saturation
            x = c * (1 - abs((hue * 6) % 2 - 1))
            m = value - c

            if hue < 1 / 6:
                rgb = (c, x, 0)
            elif hue < 2 / 6:
                rgb = (x, c, 0)
            elif hue < 3 / 6:
                rgb = (0, c, x)
            elif hue < 4 / 6:
                rgb = (0, x, c)
            elif hue < 5 / 6:
                rgb = (x, 0, c)
            else:
                rgb = (c, 0, x)

            color = tuple(int((cc + m) * 255) for cc in rgb)
            colors.append(color)
        return colors

    def generate_caption(self, pil_image):
        """Generate caption for the image"""
        try:
            # Prepare image for captioning
            inputs = self.cap_processor(pil_image, return_tensors="pt").to(self.device)

            # Generate caption
            with torch.no_grad():
                out = self.cap_model.generate(**inputs, max_length=50)
                caption = self.cap_processor.decode(out[0], skip_special_tokens=True)

            return caption
        except Exception as e:
            print(f"Caption generation error: {e}")
            return "Caption generation failed"

    def process_frame(self, frame, frame_count):
        """Process a single frame for both detection and captioning"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Update caption periodically
        if frame_count % self.caption_interval == 0:
            self.current_caption = self.generate_caption(pil_image)

        # Prepare image for detection
        inputs = self.det_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get detection predictions
        with torch.no_grad():
            outputs = self.det_model(**inputs)

        # Process detection results
        results = self.det_processor.post_process_object_detection(
            outputs,
            threshold=self.confidence_threshold,
            target_sizes=[(pil_image.height, pil_image.width)],
        )[0]

        # Draw detections and caption
        annotated_frame = frame.copy()

        # Draw caption at the top
        cv2.rectangle(annotated_frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(
            annotated_frame,
            f"Scene Description: {self.current_caption}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Draw detected objects
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            box = [int(i) for i in box.tolist()]
            score = score.item()
            label = label.item()

            color = self.colors[label % len(self.colors)]

            # Draw box
            cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), color, 2)

            # Prepare label text
            label_text = f"{self.det_model.config.id2label[label]}: {score:.2f}"

            # Calculate text size
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            # Draw label background
            cv2.rectangle(
                annotated_frame,
                (box[0], box[1] - text_height - 4),
                (box[0] + text_width, box[1]),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                annotated_frame,
                label_text,
                (box[0], box[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        return annotated_frame

    def start_realtime_detection(self, camera_index=0, display_fps=True):
        """Start real-time detection and captioning from webcam"""
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Starting real-time detection and captioning... Press 'q' to quit.")

        frame_count = 0
        fps = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            annotated_frame = self.process_frame(frame, frame_count)

            # Calculate and display FPS
            frame_count += 1
            if display_fps and frame_count % 30 == 0:
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                start_time = time.time()

            if display_fps:
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            # Display the frame
            cv2.imshow("Real-time Detection and Captioning", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    # Create detector instance
    detector = RealTimeDetectionAndCaption(
        detection_model="PekingU/rtdetr_v2_r50vd",
        caption_model="Salesforce/blip-image-captioning-large",
        confidence_threshold=0.5,
    )

    # Start real-time detection and captioning
    detector.start_realtime_detection(camera_index=0, display_fps=True)


if __name__ == "__main__":
    main()
